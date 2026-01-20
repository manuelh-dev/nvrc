// SPDX-License-Identifier: Apache-2.0
// Copyright (c) NVIDIA CORPORATION

mod daemon;
mod execute;
mod kata_agent;
mod kernel_params;
mod kmsg;
mod lockdown;
mod macros;
mod modprobe;
mod mount;
mod nvrc;
mod smi;
mod syslog;
mod toolkit;

#[cfg(all(target_os = "linux", target_env = "musl", target_arch = "x86_64"))]
use core::arch::global_asm;

// Minimal entrypoint to emit a raw write before libc startup.
// Uses syscalls only and then tail-calls into __wrap___libc_start_main.
#[cfg(all(target_os = "linux", target_env = "musl", target_arch = "x86_64"))]
global_asm!(
    r#"
    .global _nvrc_start
    .type _nvrc_start,@function
_nvrc_start:
    // write to stderr/stdout
    mov rax, 1
    mov rdi, 2
    lea rsi, [rip + _nvrc_start_msg]
    mov edx, 21
    syscall
    mov rax, 1
    mov rdi, 1
    lea rsi, [rip + _nvrc_start_msg]
    mov edx, 21
    syscall

    // openat(AT_FDCWD, "/proc/kmsg", O_WRONLY|O_CLOEXEC)
    mov rax, 257
    mov rdi, -100
    lea rsi, [rip + _nvrc_path_proc_kmsg]
    mov rdx, 0x00080001
    xor r10, r10
    syscall
    test rax, rax
    js 1f
    mov rdi, rax
    mov rax, 1
    lea rsi, [rip + _nvrc_start_msg]
    mov edx, 21
    syscall
    mov rax, 3
    syscall
1:
    // openat(AT_FDCWD, "/dev/console", O_WRONLY|O_CLOEXEC)
    mov rax, 257
    mov rdi, -100
    lea rsi, [rip + _nvrc_path_dev_console]
    mov rdx, 0x00080001
    xor r10, r10
    syscall
    test rax, rax
    js 2f
    mov rdi, rax
    mov rax, 1
    lea rsi, [rip + _nvrc_start_msg]
    mov edx, 21
    syscall
    mov rax, 3
    syscall
2:
    // openat(AT_FDCWD, "/dev/hvc0", O_WRONLY|O_CLOEXEC)
    mov rax, 257
    mov rdi, -100
    lea rsi, [rip + _nvrc_path_dev_hvc0]
    mov rdx, 0x00080001
    xor r10, r10
    syscall
    test rax, rax
    js 3f
    mov rdi, rax
    mov rax, 1
    lea rsi, [rip + _nvrc_start_msg]
    mov edx, 21
    syscall
    mov rax, 3
    syscall
3:
    // openat(AT_FDCWD, "/dev/hvc1", O_WRONLY|O_CLOEXEC)
    mov rax, 257
    mov rdi, -100
    lea rsi, [rip + _nvrc_path_dev_hvc1]
    mov rdx, 0x00080001
    xor r10, r10
    syscall
    test rax, rax
    js 4f
    mov rdi, rax
    mov rax, 1
    lea rsi, [rip + _nvrc_start_msg]
    mov edx, 21
    syscall
    mov rax, 3
    syscall
4:

    mov rsi, [rsp]
    lea rdx, [rsp + 8]
    xor rcx, rcx
    xor r8, r8
    xor r9, r9
    mov r10, rsp
    sub rsp, 8
    mov [rsp], r10
    lea rdi, [rip + main]
    call __wrap___libc_start_main
    hlt

    .section .rodata
_nvrc_start_msg:
    .ascii "NVRC _start: entered\n"
    .equ _nvrc_start_len, 21
_nvrc_path_proc_kmsg:
    .ascii "/proc/kmsg\0"
_nvrc_path_dev_console:
    .ascii "/dev/console\0"
_nvrc_path_dev_hvc0:
    .ascii "/dev/hvc0\0"
_nvrc_path_dev_hvc1:
    .ascii "/dev/hvc1\0"
    "#
);

pub use macros::ResultExt;

#[cfg(test)]
mod test_utils;

#[macro_use]
extern crate log;
extern crate kernlog;

use std::collections::HashMap;
use core::sync::atomic::{AtomicBool, Ordering};

use kata_agent::SYSLOG_POLL_FOREVER as POLL_FOREVER;
use nvrc::NVRC;
use toolkit::nvidia_ctk_cdi;

type ModeFn = fn(&mut NVRC);

#[cfg(all(target_os = "linux", target_env = "musl"))]
extern "C" {
    fn __real___libc_start_main(
        main: extern "C" fn(i32, *const *const u8, *const *const u8) -> i32,
        argc: i32,
        argv: *const *const u8,
        init: Option<extern "C" fn()>,
        fini: Option<extern "C" fn()>,
        rtld_fini: Option<extern "C" fn()>,
        stack_end: *mut libc::c_void,
    ) -> i32;
}

/// Linker wrapper to log before Rust runtime calls main().
#[cfg(all(target_os = "linux", target_env = "musl"))]
#[no_mangle]
pub extern "C" fn __wrap___libc_start_main(
    main: extern "C" fn(i32, *const *const u8, *const *const u8) -> i32,
    argc: i32,
    argv: *const *const u8,
    init: Option<extern "C" fn()>,
    fini: Option<extern "C" fn()>,
    rtld_fini: Option<extern "C" fn()>,
    stack_end: *mut libc::c_void,
) -> i32 {
    unsafe {
        let msg = b"NVRC wrap: __libc_start_main\n";
        lockdown::early_boot_log(msg);
        lockdown::run_early_init_for_wrapper();
        __real___libc_start_main(main, argc, argv, init, fini, rtld_fini, stack_end)
    }
}

// Prevent LTO from discarding the wrapper symbol.
#[cfg(all(target_os = "linux", target_env = "musl"))]
#[used]
static WRAP_FORCE: extern "C" fn(
    extern "C" fn(i32, *const *const u8, *const *const u8) -> i32,
    i32,
    *const *const u8,
    Option<extern "C" fn()>,
    Option<extern "C" fn()>,
    Option<extern "C" fn()>,
    *mut libc::c_void,
) -> i32 = __wrap___libc_start_main;

#[cfg(all(target_os = "linux", target_env = "musl"))]
extern "C" {
    fn __real_sysconf(name: libc::c_int) -> libc::c_long;
}

#[cfg(all(target_os = "linux", target_env = "musl"))]
static SYSCONF_ALLOW_REAL: AtomicBool = AtomicBool::new(false);

/// Trace sysconf calls during early runtime init.
#[cfg(all(target_os = "linux", target_env = "musl"))]
#[no_mangle]
pub extern "C" fn __wrap_sysconf(name: libc::c_int) -> libc::c_long {
    unsafe {
        lockdown::early_boot_log_num(b"NVRC wrap: sysconf ", name as i64);
        if SYSCONF_ALLOW_REAL.load(Ordering::Relaxed) {
            return __real_sysconf(name);
        }
        // Minimal stubs to avoid early musl aborts during startup.
        match name {
            libc::_SC_PAGE_SIZE => 4096,
            libc::_SC_NPROCESSORS_ONLN | libc::_SC_NPROCESSORS_CONF => 1,
            libc::_SC_CLK_TCK => 100,
            libc::_SC_PHYS_PAGES | libc::_SC_AVPHYS_PAGES => 0,
            _ => -1,
        }
    }
}

// Prevent LTO from discarding the sysconf wrapper symbol.
#[cfg(all(target_os = "linux", target_env = "musl"))]
#[used]
static WRAP_SYSCONF_FORCE: extern "C" fn(libc::c_int) -> libc::c_long = __wrap_sysconf;

#[cfg(all(target_os = "linux", target_env = "musl"))]
extern "C" {
    fn __real_abort() -> !;
}

/// Trace aborts during early runtime init.
#[cfg(all(target_os = "linux", target_env = "musl"))]
#[no_mangle]
pub extern "C" fn __wrap_abort() -> ! {
    unsafe {
        let msg = b"NVRC wrap: abort\n";
        lockdown::early_boot_log(msg);
        __real_abort()
    }
}

// Prevent LTO from discarding the abort wrapper symbol.
#[cfg(all(target_os = "linux", target_env = "musl"))]
#[used]
static WRAP_ABORT_FORCE: extern "C" fn() -> ! = __wrap_abort;

/// VMs with GPU passthrough need driver setup, clock tuning,
/// and monitoring daemons before workloads can use the GPU.
fn mode_gpu(init: &mut NVRC) {
    modprobe::load("nvidia");
    modprobe::load("nvidia-uvm");

    init.nvidia_smi_lmc();
    init.nvidia_smi_lgc();
    init.nvidia_smi_pl();

    init.nvidia_persistenced();

    init.nv_hostengine();
    init.dcgm_exporter();
    init.nv_fabricmanager();
    nvidia_ctk_cdi();
    init.nvidia_smi_srs();
    init.check_daemons();
}

/// NVSwitch NVL4 mode for HGX H100/H200/H800 systems (third-gen NVSwitch).
/// Service VM mode for NVLink 4.0 topologies in shared virtualization.
/// Loads NVIDIA driver and starts fabric manager. GPUs are assigned to service VM.
/// Automatically enables fabricmanager regardless of kernel parameters.
fn mode_nvswitch_nvl4(init: &mut NVRC) {
    // Override kernel parameter: always enable fabricmanager for nvswitch mode
    init.fabricmanager_enabled = Some(true);

    modprobe::load("nvidia");
    init.nv_fabricmanager();
    init.check_daemons();
}

/// NVSwitch NVL5 mode for HGX B200/B300/B100 systems (fourth-gen NVSwitch).
/// Service VM mode for NVLink 5.0 topologies with CX7 bridge devices.
/// Does NOT load nvidia driver (GPUs not attached to service VM).
/// Loads ib_umad for InfiniBand MAD access to CX7 bridges.
/// FM automatically starts NVLSM (NVLink Subnet Manager) internally.
/// Requires kernel 5.17+ and /dev/infiniband/umadX devices.
fn mode_nvswitch_nvl5(init: &mut NVRC) {
    // Override kernel parameter: always enable fabricmanager for nvswitch mode
    init.fabricmanager_enabled = Some(true);

    // Load InfiniBand user MAD module for CX7 bridge device access
    modprobe::load("ib_umad");
    init.nv_fabricmanager();
    init.check_daemons();
}

fn main() {
    // Dispatch table allows adding new modes without touching control flow.
    let modes: HashMap<&str, ModeFn> = HashMap::from([
        ("gpu", mode_gpu as ModeFn),
        ("cpu", (|_| {}) as ModeFn),
        ("nvswitch-nvl4", mode_nvswitch_nvl4 as ModeFn),
        ("nvswitch-nvl5", mode_nvswitch_nvl5 as ModeFn),
    ]);

    SYSCONF_ALLOW_REAL.store(true, Ordering::SeqCst);
    lockdown::set_signal_handlers();
    lockdown::set_panic_hook();
    let mut init = NVRC::default();
    mount::setup();
    kmsg::kernlog_setup();
    syslog::poll();
    mount::readonly("/");
    init.process_kernel_params(None);

    // Kernel param nvrc.mode selects runtime behavior; GPU is the safe default
    // since most users expect full GPU functionality.
    let mode = init.mode.as_deref().unwrap_or("gpu");
    let setup = modes.get(mode).copied().unwrap_or(mode_gpu);
    setup(&mut init);

    lockdown::disable_modules_loading();
    kata_agent::fork_agent(POLL_FOREVER);
}
