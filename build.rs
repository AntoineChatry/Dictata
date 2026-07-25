//! Embeds the application icon and the version metadata into the Windows exe.
//!
//! Without this, `dictata.exe` shows the generic Windows icon in Explorer, the
//! taskbar and Task Manager, and its Properties dialog has no Details tab —
//! which also makes the binary look untrustworthy next to a SmartScreen prompt.
//!
//! Requires `rc.exe` from the Windows SDK, which the MSVC toolchain already
//! needs in order to link at all. If it is missing anyway, the build emits a
//! warning and continues: a missing icon must not stop someone from compiling
//! and running the application. The release script (`scripts/release-windows.ps1`)
//! asserts that the metadata really is present, so a release cannot silently
//! ship without it.

fn main() {
    // Rebuild when the icon changes; without this cargo would keep the stale
    // resource from the previous build.
    println!("cargo::rerun-if-changed=assets/dictata.ico");
    println!("cargo::rerun-if-changed=build.rs");

    // Target, not host: this must not fire when cross-compiling to Linux.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }

    let mut res = winresource::WindowsResource::new();
    // FileVersion, ProductVersion, FILEVERSION and PRODUCTVERSION are already
    // filled from CARGO_PKG_VERSION by `new()`, so the manifest stays the
    // single source of truth for the version number. Only the human-readable
    // strings are set here; they are set programmatically rather than through
    // `[package.metadata.winresource]` because reading that section depends on
    // an optional `toml` feature of the crate.
    res.set_icon("assets/dictata.ico")
        // Windows uses FileDescription as the application name in Task Manager
        // and in dialogs — not ProductName.
        .set("FileDescription", "Dictata - local voice dictation")
        .set("ProductName", "Dictata")
        .set("OriginalFilename", "dictata.exe")
        .set("InternalName", "dictata")
        .set("CompanyName", "Antoine Chatry")
        .set(
            "LegalCopyright",
            "Copyright (c) 2026 Antoine Chatry - MIT License with Commons Clause",
        )
        .set(
            "Comments",
            "100% local voice dictation. No data ever leaves the machine.",
        );

    if let Err(e) = res.compile() {
        println!("cargo::warning=icon and version metadata not embedded: {e}");
    }
}
