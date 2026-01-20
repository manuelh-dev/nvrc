// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

//! Lockdown primitives for confidential VM security.
//!
//! In production, panic triggers VM power-off. For tests, the shutdown
//! action is configurable via `set_panic_hook_with()`.

use crate::macros::ResultExt;
use nix::sys::reboot::{reboot, RebootMode};
use nix::sys::signal::{sigaction, SaFlags, SigAction, SigHandler, SigSet, Signal};
use nix::unistd::sync;
use std::fs;
use std::mem;
use std::panic;

/// Default shutdown action: power off the VM.
fn power_off() {
    let _ = reboot(RebootMode::RB_POWER_OFF);
}

/// Install a panic handler that powers off the VM instead of unwinding.
/// In a confidential VM, a panic could leave the system in an undefined state
/// with potential data exposure. Power-off ensures clean termination—the host
/// hypervisor will see the VM exit and can handle cleanup appropriately.
/// sync() flushes pending writes before power-off to preserve any logs.
pub fn set_panic_hook() {
    set_panic_hook_with(power_off)
}

#[cfg(target_arch = "x86_64")]
unsafe fn raw_write(fd: i64, buf: *const u8, len: usize) {
    use core::arch::asm;
    asm!(
        "syscall",
        in("rax") 1_i64, // SYS_write
        in("rdi") fd,
        in("rsi") buf,
        in("rdx") len,
        lateout("rcx") _,
        lateout("r11") _,
    );
}

#[cfg(target_arch = "x86_64")]
unsafe fn raw_openat(path: *const u8, flags: i64) -> i64 {
    use core::arch::asm;
    let ret: i64;
    asm!(
        "syscall",
        in("rax") 257_i64, // SYS_openat
        in("rdi") -100_i64, // AT_FDCWD
        in("rsi") path,
        in("rdx") flags,
        in("r10") 0_i64, // mode
        lateout("rax") ret,
        lateout("rcx") _,
        lateout("r11") _,
    );
    ret
}

#[cfg(target_arch = "x86_64")]
unsafe fn raw_close(fd: i64) {
    use core::arch::asm;
    asm!(
        "syscall",
        in("rax") 3_i64, // SYS_close
        in("rdi") fd,
        lateout("rcx") _,
        lateout("r11") _,
    );
}

#[cfg(target_arch = "x86_64")]
unsafe fn raw_write_to_optional_sinks(msg: &[u8]) {
    let flags = (libc::O_WRONLY | libc::O_CLOEXEC) as i64;
    for path in [
        b"/dev/console\0".as_slice(),
        b"/dev/kmsg\0".as_slice(),
        b"/dev/ttyS0\0".as_slice(),
        b"/dev/ttyS1\0".as_slice(),
        b"/dev/hvc0\0".as_slice(),
        b"/proc/kmsg\0".as_slice(),
        b"/proc/self/fd/1\0".as_slice(),
        b"/proc/self/fd/2\0".as_slice(),
    ] {
        let fd = raw_openat(path.as_ptr(), flags);
        if fd >= 0 {
            raw_write(fd, msg.as_ptr(), msg.len());
            raw_close(fd);
        }
    }
}

pub(crate) unsafe fn early_boot_log(msg: &[u8]) {
    // Avoid libc in early boot: TLS may not be initialized yet.
    #[cfg(target_arch = "x86_64")]
    {
        raw_write(libc::STDOUT_FILENO as i64, msg.as_ptr(), msg.len());
        raw_write(libc::STDERR_FILENO as i64, msg.as_ptr(), msg.len());
        raw_write_to_optional_sinks(msg);
    }
}

#[allow(dead_code)]
pub(crate) unsafe fn early_boot_log_num(prefix: &[u8], num: i64) {
    // Build: "<prefix><num>\n" with no libc usage.
    #[cfg(target_arch = "x86_64")]
    {
        let mut buf = [0u8; 64];
        let mut idx = 0usize;
        for &b in prefix {
            if idx >= buf.len() {
                break;
            }
            buf[idx] = b;
            idx += 1;
        }

        let mut n = num;
        if n == 0 {
            if idx < buf.len() {
                buf[idx] = b'0';
                idx += 1;
            }
        } else {
            if n < 0 {
                if idx < buf.len() {
                    buf[idx] = b'-';
                    idx += 1;
                }
                n = -n;
            }
            let mut digits = [0u8; 20];
            let mut dlen = 0usize;
            while n > 0 && dlen < digits.len() {
                digits[dlen] = b'0' + (n % 10) as u8;
                n /= 10;
                dlen += 1;
            }
            while dlen > 0 && idx < buf.len() {
                dlen -= 1;
                buf[idx] = digits[dlen];
                idx += 1;
            }
        }

        if idx < buf.len() {
            buf[idx] = b'\n';
            idx += 1;
        }

        raw_write(libc::STDOUT_FILENO as i64, buf.as_ptr(), idx);
        raw_write(libc::STDERR_FILENO as i64, buf.as_ptr(), idx);
        raw_write_to_optional_sinks(&buf[..idx]);
    }
}

extern "C" fn fatal_signal_handler(signum: libc::c_int) {
    let msg: &[u8] = match signum {
        libc::SIGSEGV => b"NVRC fatal signal: SIGSEGV\n",
        libc::SIGILL => b"NVRC fatal signal: SIGILL\n",
        libc::SIGBUS => b"NVRC fatal signal: SIGBUS\n",
        libc::SIGFPE => b"NVRC fatal signal: SIGFPE\n",
        libc::SIGABRT => b"NVRC fatal signal: SIGABRT\n",
        _ => b"NVRC fatal signal: UNKNOWN\n",
    };
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            raw_write(libc::STDERR_FILENO as i64, msg.as_ptr(), msg.len());
            // Best-effort additional sinks: async-signal-safe syscalls only.
            raw_write_to_optional_sinks(msg);
        }
        libc::_exit(128 + signum);
    }
}

unsafe extern "C" fn early_init() {
    // Best-effort early marker for visibility before main().
    let msg = b"NVRC early init: installing signal handlers\n";
    early_boot_log(msg);

    let mut action: libc::sigaction = mem::zeroed();
    action.sa_flags = libc::SA_RESETHAND | libc::SA_NODEFER;
    action.sa_sigaction = fatal_signal_handler as usize;
    libc::sigemptyset(&mut action.sa_mask);

    for signum in [
        libc::SIGSEGV,
        libc::SIGILL,
        libc::SIGBUS,
        libc::SIGFPE,
        libc::SIGABRT,
    ] {
        let _ = libc::sigaction(signum, &action, std::ptr::null_mut());
    }
}

// Run before main() to catch crashes during early runtime init.
#[used]
#[cfg_attr(target_os = "linux", link_section = ".preinit_array")]
static EARLY_INIT: unsafe extern "C" fn() = early_init;

#[allow(dead_code)]
pub(crate) unsafe fn run_early_init_for_wrapper() {
    early_init();
}

/// Install signal handlers for fatal crashes that bypass Rust panics.
/// These handlers emit a minimal message to stderr and exit with a signal code,
/// which helps identify early failures (e.g., SIGSEGV/SIGILL) in init.
pub fn set_signal_handlers() {
    let action = SigAction::new(
        SigHandler::Handler(fatal_signal_handler),
        SaFlags::SA_RESETHAND | SaFlags::SA_NODEFER,
        SigSet::empty(),
    );
    for signal in [
        Signal::SIGSEGV,
        Signal::SIGILL,
        Signal::SIGBUS,
        Signal::SIGFPE,
        Signal::SIGABRT,
    ] {
        if let Err(err) = unsafe { sigaction(signal, &action) } {
            panic!("install signal handler {signal}: {err}");
        }
    }
}

/// Internal: panic handler with configurable shutdown (for unit tests).
/// Production uses power_off(); tests inject a no-op to avoid rebooting.
fn set_panic_hook_with<F: Fn() + Send + Sync + 'static>(shutdown: F) {
    panic::set_hook(Box::new(move |panic_info| {
        let msg = format!("panic (new): {panic_info}");
        // Try all available outputs - some may not exist yet during early init
        // stderr/stdout: always available (fd 1,2 from kernel)
        eprintln!("panic (original): {panic_info}");
        eprintln!("eprintl: {msg}");
        println!("println: {msg}");
        // /dev/console: may exist from initramfs before devtmpfs mount
        let _ = fs::write("/dev/console", format!("dev-console: {msg}\n"));
        // /dev/kmsg: only after devtmpfs mounted, <0> = KERN_EMERG
        let _ = fs::write("/dev/kmsg", format!("<0> dev-kmsg: {msg}\n"));
        // Logger: only after kernlog_setup()
        log::error!("logger-error: {msg}");
        sync();
        shutdown();
    }));
}

/// Permanently disable kernel module loading for this boot.
/// Once all required GPU drivers are loaded, this prevents any further
/// module insertion—a security hardening measure for confidential VMs
/// that blocks potential kernel-level attacks via malicious modules.
/// This is a one-way operation: once set, it cannot be undone without reboot.
pub fn disable_modules_loading() {
    const PATH: &str = "/proc/sys/kernel/modules_disabled";
    fs::write(PATH, b"1\n").or_panic(format_args!("disable module loading {PATH}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::require_root;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_set_panic_hook_with_custom_action() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        // Install hook with test closure
        set_panic_hook_with(move || {
            called_clone.store(true, Ordering::SeqCst);
        });

        // The hook is installed - we can't trigger it without panicking,
        // but we've exercised the code path
        assert!(!called.load(Ordering::SeqCst)); // Not called yet
    }

    #[test]
    #[ignore] // Permanently disables module loading until reboot - run with --include-ignored on CI
    fn test_disable_modules_loading() {
        require_root();

        // This permanently disables module loading until reboot.
        // Only run on dedicated test runners!
        disable_modules_loading();

        // Verify it was set
        let content = fs::read_to_string("/proc/sys/kernel/modules_disabled").unwrap();
        assert_eq!(content.trim(), "1");
    }

    #[test]
    fn test_power_off_function_exists() {
        // Just verify power_off compiles - can't call it without rebooting!
        let _: fn() = power_off;
    }

    #[test]
    #[ignore] // Installs real power_off hook - run with --include-ignored on CI
    fn test_set_panic_hook() {
        // Installs the real hook (with power_off) - just don't trigger it!
        set_panic_hook();
        // If we got here, the hook was installed successfully
    }
}
