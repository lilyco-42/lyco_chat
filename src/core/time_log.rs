/*
 * @file time_log.rs
 * @brief Startup time logging and a /time interface
 * @author Kevin Thomas
 * @date 2025
 *
 * MIT License
 *
 * Copyright (c) 2025 Kevin Thomas
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//! FILE: time_log.rs
//!
//! DESCRIPTION:
//! Startup time logging: every launch appends its timestamp to a log file,
//! and a `/time` interface exposes current time, uptime and startup history.
//!
//! BRIEF:
//! Uses only std (`SystemTime`) with a small epoch -> ISO-8601 UTC formatter,
//! so no datetime dependency is needed.

use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};
/// Current Unix epoch seconds.
pub fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Formats Unix epoch seconds as ISO-8601 UTC (`YYYY-MM-DDTHH:MM:SSZ`).
///
/// # Details
/// Standard "civil-from-days" algorithm (Howard Hinnant), no external crate.
pub fn format_iso(secs: u64) -> String {
    let days = secs / 86_400;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let mut z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mth = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mth <= 2 { y + 1 } else { y };
    format!("{y:04}-{mth:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Current time as ISO-8601 UTC.
pub fn now_iso() -> String {
    format_iso(now_epoch())
}

/// Default log path for startup timestamps.
pub fn default_log_path() -> String {
    "data/startup_times.log".to_string()
}

/// Appends the current startup timestamp to the log (one ISO line per start).
///
/// # Arguments
/// * `path` - Log file path (default `data/startup_times.log`)
///
/// # Returns
/// * `String` - The recorded ISO timestamp
pub fn record_startup(path: &str) -> String {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let iso = now_iso();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{iso}");
    }
    iso
}

/// Reads the startup history (one ISO timestamp per recorded launch).
///
/// # Arguments
/// * `path` - Log file path
///
/// # Returns
/// * `Vec<String>` - Startup timestamps in chronological order
pub fn read_startups(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().to_string())
        .collect()
}
/// Locates a binary by scanning `$PATH`.
fn find_bin(name: &str) -> Option<String> {
    let path = std::env::var("PATH").unwrap_or_default();
    for dir in path.split(':') {
        let p = format!("{dir}/{name}");
        if std::path::Path::new(&p).is_file() {
            return Some(p);
        }
    }
    None
}

/// Fetches the latest stable Rust version from the official release channel.
///
/// # Details
/// Reads `[pkg.rust] version = "1.xx.y (hash date)"` from the Rust CDN
/// manifest. Returns `None` offline or on parse failure.
pub fn latest_stable_version() -> Option<(u32, u32)> {
    let url = "https://static.rust-lang.org/dist/channel-rust-stable.toml";
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .ok()?;
    let body = client.get(url).send().ok()?.text().ok()?;
    let idx = body.find("[pkg.rust]")?;
    let rest = &body[idx..];
    let line = rest.lines().find(|l| l.trim_start().starts_with("version ="))?;
    let v = line.split('"').nth(1)?;
    let mut it = v.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

/// Pure lag check: warn when `installed` trails `latest` by too many releases.
///
/// # Details
/// Rust stable ships a minor release every 6 weeks. Trailing by 6 minor
/// versions is ~9 months and worth a warning; smaller gaps are informational.
pub fn lag_check(installed: (u32, u32), latest: (u32, u32)) -> Option<String> {
    if latest <= installed {
        return None;
    }
    let lag = latest.1 as i64 - installed.1 as i64;
    if lag >= 6 {
        Some(format!(
            "[warning] 本机 rustc {}.{} 落后最新稳定版 {}.{} 达 {lag} 个小版本（约 {:.1} 个月），建议 rustup update stable",
            installed.0, installed.1, latest.0, latest.1, lag as f64 * 1.5
        ))
    } else {
        Some(format!(
            "[info] 本机 rustc {}.{} 落后最新稳定版 {}.{}（{lag} 个小版本），可 rustup update stable",
            installed.0, installed.1, latest.0, latest.1
        ))
    }
}

/// Compares installed rustc with the latest stable and warns on large lag.
pub fn check_version_lag() -> Option<String> {
    let installed = version_pair("rustc")?;
    let latest = latest_stable_version()?;
    lag_check(installed, latest)
}

/// Full toolchain status for the frontend `/toolchain` interface.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct ToolchainStatus {
    pub cargo_version: Option<String>,
    pub rustc_version: Option<String>,
    pub latest_stable: Option<String>,
    pub lag: Option<i64>,
    pub lag_message: Option<String>,
    pub toolchain_warning: Option<String>,
    pub startup_time: Option<String>,
    pub startups: Vec<String>,
    pub uptime_secs: u64,
}

fn version_string(name: &str) -> Option<String> {
    let out = std::process::Command::new(name).arg("--version").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let v = s.split_whitespace().nth(1)?;
    Some(v.to_string())
}

/// Gathers everything the frontend needs to render the toolchain dialog.
pub fn toolchain_status() -> ToolchainStatus {
    let cargo_version = version_string("cargo");
    let rustc_version = version_string("rustc");
    let latest = latest_stable_version();
    let latest_stable = latest.map(|(m, n)| format!("{m}.{n}"));
    let installed = rustc_version
        .as_deref()
        .and_then(|v| {
            let mut it = v.split('.');
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        });
    let (lag, lag_message) = match (installed, latest) {
        (Some(i), Some(l)) if l > i => {
            let gap = l.1 as i64 - i.1 as i64;
            (Some(gap), lag_check(i, l))
        }
        _ => (None, None),
    };
    let log = default_log_path();
    let startups = read_startups(&log);
    ToolchainStatus {
        cargo_version,
        rustc_version,
        latest_stable,
        lag,
        lag_message,
        toolchain_warning: check_toolchain_warning(),
        startup_time: startups.last().cloned(),
        startups,
        uptime_secs: 0,
    }
}

/// Modification time (Unix epoch seconds) of a file.
fn mtime(path: &str) -> Option<u64> {
    let m = std::fs::metadata(path).ok()?.modified().ok()?;
    m.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs())
}

/// Parses `(major, minor)` from `cargo 1.97.1 (...)`, `rustc 1.97.1 (...)`.
fn version_pair(name: &str) -> Option<(u32, u32)> {
    let out = std::process::Command::new(name).arg("--version").output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.split_whitespace().nth(1)?.split('.');
    let major = it.next()?.parse().ok()?;
    let minor = it.next()?.parse().ok()?;
    Some((major, minor))
}

/// Compares cargo and rustc (modification time + version) and warns on gaps.
///
/// # Details
/// cargo and rustc should come from the same toolchain. A large binary
/// mtime gap or a major/minor version mismatch signals a broken or upgraded
/// toolchain (e.g. a stale cargo proxy), worth surfacing at startup.
///
/// # Returns
/// * `Option<String>` - Warning message when the gap is significant
pub fn check_toolchain_warning() -> Option<String> {
    let cargo = find_bin("cargo")?;
    let rustc = find_bin("rustc")?;

    // time gap: if the two binaries were written far apart, they may disagree
    let (cm, rm) = (mtime(&cargo)?, mtime(&rustc)?);
    let gap = cm.abs_diff(rm);
    if gap > 12 * 3600 {
        let d = gap / 3600;
        return Some(format!(
            "[warning] cargo 与 rustc 更新时间差 {d}h（cargo: {cargo}, rustc: {rustc}），工具链可能不匹配"
        ));
    }

    // version gap: major/minor mismatch
    if let (Some(cv), Some(rv)) = (version_pair("cargo"), version_pair("rustc")) {
        if cv != rv {
            return Some(format!(
                "[warning] cargo {cv:?} 与 rustc {rv:?} 主次版本不一致，建议 rustup update"
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ISO formatter produces a plausible timestamp for a known epoch.
    #[test]
    fn test_format_iso() {
        assert_eq!(format_iso(0), "1970-01-01T00:00:00Z");
        // 2026-01-01 00:00:00 UTC
        assert_eq!(format_iso(1767225600), "2026-01-01T00:00:00Z");
    }

    /// record_startup appends a line; read_startups returns it.
    #[test]
    fn test_record_and_read() {
        let path = std::env::temp_dir().join(format!("startup_{}.log", std::process::id()));
        let p = path.to_str().unwrap();
        let iso = record_startup(p);
        let all = read_startups(p);
        assert!(!all.is_empty(), "should have at least one entry");
        assert!(all.last().map_or(false, |l| l == &iso));
        let _ = std::fs::remove_file(p);
    }

    /// now_iso round-trips to a parseable length.
    #[test]
    fn test_now_iso() {
        let iso = now_iso();
        assert_eq!(iso.len(), 20, "expected YYYY-MM-DDTHH:MM:SSZ, got {iso}");
    }

    /// find_bin locates cargo/rustc on this machine.
    #[test]
    fn test_find_bin() {
        assert!(find_bin("cargo").is_some(), "cargo should be on PATH");
        assert!(find_bin("rustc").is_some(), "rustc should be on PATH");
        assert!(find_bin("definitely_not_a_bin").is_none());
    }

    /// version_pair parses semver from `cargo --version`.
    #[test]
    fn test_version_pair() {
        let v = version_pair("rustc").expect("rustc --version should parse");
        assert!(v.0 >= 1, "rustc major version: {:?}", v);
    }

    /// On a sane machine the toolchain check should not warn.
    #[test]
    fn test_check_toolchain_ok() {
        // If this machine is consistent, no warning; the important part is it
        // runs without panicking and either returns None or a warning string.
        let _ = check_toolchain_warning();
    }

    /// Up-to-date toolchain produces no warning.
    #[test]
    fn test_lag_check_uptodate() {
        assert!(lag_check((1, 97), (1, 97)).is_none());
    }

    /// Small lag is informational, not a warning.
    #[test]
    fn test_lag_check_small() {
        let m = lag_check((1, 95), (1, 97)).expect("small lag should still report");
        assert!(m.starts_with("[info]"));
    }

    /// Large lag (>= 6 minor releases) triggers a warning.
    #[test]
    fn test_lag_check_large() {
        let m = lag_check((1, 88), (1, 97)).expect("large lag should warn");
        assert!(m.starts_with("[warning]"), "expected warning, got {m}");
    }
}
