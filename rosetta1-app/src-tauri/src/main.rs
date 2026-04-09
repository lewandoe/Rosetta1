// Suppress the Windows console window in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};

// ─────────────────────────────────────────────────────────────────────────────
// Known paths — repo is always at /Users/eric/Rosetta1
// ─────────────────────────────────────────────────────────────────────────────
const REPO: &str = "/Users/eric/Rosetta1";

fn repo()    -> PathBuf { PathBuf::from(REPO) }
fn db_path() -> PathBuf { repo().join("db").join("trades.db") }
fn env_path()-> PathBuf { repo().join(".env") }
fn py_exe()  -> PathBuf { repo().join(".venv").join("bin").join("python") }
fn cfg_path()-> PathBuf { repo().join("config").join("settings.py") }

// ─────────────────────────────────────────────────────────────────────────────
// Managed state — the running child process
// ─────────────────────────────────────────────────────────────────────────────
struct BotProcess {
    child: Mutex<Option<Child>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialisable types returned to the frontend
// ─────────────────────────────────────────────────────────────────────────────
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Trade {
    symbol:       String,
    signal_type:  Option<String>,
    opened_at:    Option<String>,
    closed_at:    Option<String>,
    entry_price:  Option<f64>,
    exit_price:   Option<f64>,
    shares:       Option<f64>,
    gross_pnl:    Option<f64>,
    hold_seconds: Option<f64>,
    confidence:   Option<f64>,
    exit_reason:  Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Position {
    symbol:      String,
    signal_type: Option<String>,
    opened_at:   Option<String>,
    entry_price: Option<f64>,
    shares:      Option<f64>,
    confidence:  Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AccountSummary {
    today_pnl:      f64,
    trade_count:    i64,
    win_count:      i64,
    win_rate:       f64,
    open_positions: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct PythonCheck {
    venv_found: bool,
    env_found:  bool,
    db_found:   bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// SQLite helper
// ─────────────────────────────────────────────────────────────────────────────
fn open_db() -> Result<rusqlite::Connection, String> {
    let p = db_path();
    rusqlite::Connection::open(&p)
        .map_err(|e| format!("Cannot open DB at {}: {}", p.display(), e))
}

// ─────────────────────────────────────────────────────────────────────────────
// .env helpers
// ─────────────────────────────────────────────────────────────────────────────
fn read_env_var(key: &str) -> Option<String> {
    let text = std::fs::read_to_string(env_path()).ok()?;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() { continue; }
        if let Some(eq) = line.find('=') {
            if line[..eq].trim() == key {
                let v = line[eq + 1..].trim().trim_matches('"').trim_matches('\'');
                return Some(v.to_string());
            }
        }
    }
    None
}

fn write_env_var(key: &str, value: &str) -> Result<(), String> {
    let path     = env_path();
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let mut lines: Vec<String> = existing.lines().map(|l| l.to_string()).collect();
    let mut found = false;

    for line in &mut lines {
        let t = line.trim();
        if t.starts_with('#') || t.is_empty() { continue; }
        if let Some(eq) = t.find('=') {
            if t[..eq].trim() == key {
                *line = format!("{}={}", key, value);
                found = true;
                break;
            }
        }
    }
    if !found { lines.push(format!("{}={}", key, value)); }

    std::fs::write(&path, lines.join("\n") + "\n")
        .map_err(|e| format!("Cannot write .env: {}", e))
}

// ─────────────────────────────────────────────────────────────────────────────
// settings.py parser — extracts Field(default=...) values
// ─────────────────────────────────────────────────────────────────────────────
fn parse_py_field(content: &str, field_name: &str) -> Option<String> {
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with(field_name) && t.contains("Field(default=") {
            let start = t.find("default=")? + 8;
            let rest  = &t[start..];
            if rest.starts_with('"') || rest.starts_with('\'') {
                let q   = rest.chars().next()?;
                let end = rest[1..].find(q)?;
                return Some(rest[1..end + 1].to_string());
            }
            let end = rest.find(|c: char| c == ',' || c == ')').unwrap_or(rest.len());
            return Some(rest[..end].trim().to_string());
        }
    }
    None
}

fn load_settings_json() -> Value {
    let content = std::fs::read_to_string(cfg_path()).unwrap_or_default();

    // Merge: .env override wins, then settings.py default, then hard fallback.
    let get = |py_key: &str, env_key: &str, fallback: &str| -> String {
        read_env_var(env_key)
            .or_else(|| parse_py_field(&content, py_key))
            .unwrap_or_else(|| fallback.to_string())
    };

    json!({
        "max_daily_loss":            get("max_daily_loss",            "MAX_DAILY_LOSS",            "200.0"),
        "max_open_positions":        get("max_open_positions",        "MAX_OPEN_POSITIONS",        "3"),
        "max_capital_per_trade_pct": get("max_capital_per_trade_pct", "MAX_CAPITAL_PER_TRADE_PCT", "0.10"),
        "max_stop_distance_pct":     get("max_stop_distance_pct",     "MAX_STOP_DISTANCE_PCT",     "0.02"),
        "min_confidence_score":      get("min_confidence_score",      "MIN_CONFIDENCE_SCORE",      "70"),
        "eod_liquidation_hour":      get("eod_liquidation_hour",      "EOD_LIQUIDATION_HOUR",      "15"),
        "eod_liquidation_minute":    get("eod_liquidation_minute",    "EOD_LIQUIDATION_MINUTE",    "45"),
        "log_level":                 get("log_level",                 "LOG_LEVEL",                 "INFO"),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 1 — start_bot
// ─────────────────────────────────────────────────────────────────────────────
/// Spawns `python main.py --no-dashboard [--live] [--symbols ...] [--log-level ...]`
/// using the repo's venv.  Stdout and stderr are piped; each line is emitted as
/// a "bot_log" event to the frontend.
#[tauri::command]
async fn start_bot(
    app:       AppHandle,
    state:     State<'_, BotProcess>,
    mode:      String,
    symbols:   Vec<String>,
    log_level: String,
) -> Result<(), String> {
    let mut lock = state.child.lock().unwrap();
    if lock.is_some() {
        return Err("Bot is already running".into());
    }

    let python = py_exe();
    if !python.exists() {
        return Err(format!(
            "Python not found at {}.\n\
             Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt",
            python.display()
        ));
    }

    let mut cmd = Command::new(&python);
    cmd.current_dir(repo());
    cmd.arg("main.py");
    cmd.arg("--no-dashboard");
    cmd.arg("--log-level").arg(&log_level);
    if mode == "live" { cmd.arg("--live"); }
    if !symbols.is_empty() {
        cmd.arg("--symbols");
        for s in &symbols { cmd.arg(s); }
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn Python: {}", e))?;

    // Stream stdout → bot_log events
    let stdout = child.stdout.take().ok_or("No stdout handle")?;
    let app_out = app.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if let Ok(l) = line { let _ = app_out.emit("bot_log", l); }
        }
    });

    // Stream stderr → bot_log events (prefixed so UI can tell them apart)
    let stderr = child.stderr.take().ok_or("No stderr handle")?;
    let app_err = app.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            if let Ok(l) = line { let _ = app_err.emit("bot_log", format!("[stderr] {}", l)); }
        }
    });

    *lock = Some(child);
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 2 — stop_bot
// ─────────────────────────────────────────────────────────────────────────────
/// Sends SIGTERM, waits up to 10 s, then force-kills.
#[tauri::command]
async fn stop_bot(state: State<'_, BotProcess>) -> Result<(), String> {
    let mut lock = state.child.lock().unwrap();
    if let Some(mut child) = lock.take() {
        let pid = child.id();
        #[cfg(unix)]
        { let _ = Command::new("kill").args(["-TERM", &pid.to_string()]).output(); }

        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match child.try_wait() {
                Ok(Some(_))                             => break,
                Ok(None) if Instant::now() >= deadline  => { let _ = child.kill(); break; }
                Ok(None) => std::thread::sleep(Duration::from_millis(250)),
                Err(_)   => break,
            }
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 3 — get_trades
// ─────────────────────────────────────────────────────────────────────────────
/// Returns all *closed* trades for the given ISO date (YYYY-MM-DD).
#[tauri::command]
fn get_trades(date: String) -> Result<Vec<Trade>, String> {
    let conn = open_db()?;
    let mut stmt = conn.prepare(
        "SELECT symbol, signal_type, opened_at, closed_at, entry_price, \
                exit_price, shares, gross_pnl, hold_seconds, \
                confidence, exit_reason \
         FROM trades \
         WHERE date(opened_at) = ?1 AND closed_at IS NOT NULL \
         ORDER BY opened_at ASC",
    ).map_err(|e| e.to_string())?;

    let rows = stmt.query_map([&date], |r| Ok(Trade {
        symbol:       r.get(0)?,
        signal_type:  r.get(1)?,
        opened_at:    r.get(2)?,
        closed_at:    r.get(3)?,
        entry_price:  r.get(4)?,
        exit_price:   r.get(5)?,
        shares:       r.get(6)?,
        gross_pnl:    r.get(7)?,
        hold_seconds: r.get(8)?,
        confidence:   r.get(9)?,
        exit_reason:  r.get(10)?,
    })).map_err(|e| e.to_string())?;

    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 4 — get_open_positions
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn get_open_positions() -> Result<Vec<Position>, String> {
    let conn = open_db()?;
    let mut stmt = conn.prepare(
        "SELECT symbol, signal_type, opened_at, entry_price, shares, confidence \
         FROM trades WHERE closed_at IS NULL ORDER BY opened_at ASC",
    ).map_err(|e| e.to_string())?;

    let rows = stmt.query_map([], |r| Ok(Position {
        symbol:      r.get(0)?,
        signal_type: r.get(1)?,
        opened_at:   r.get(2)?,
        entry_price: r.get(3)?,
        shares:      r.get(4)?,
        confidence:  r.get(5)?,
    })).map_err(|e| e.to_string())?;

    Ok(rows.filter_map(|r| r.ok()).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 5 — get_account_summary
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn get_account_summary() -> Result<AccountSummary, String> {
    let conn = open_db()?;

    let (today_pnl, trade_count, win_count): (f64, i64, i64) = conn.query_row(
        "SELECT COALESCE(SUM(gross_pnl),0.0), COUNT(*), \
                SUM(CASE WHEN gross_pnl > 0 THEN 1 ELSE 0 END) \
         FROM trades \
         WHERE date(opened_at) = date('now','localtime') AND closed_at IS NOT NULL",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).map_err(|e| e.to_string())?;

    let open_positions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM trades WHERE closed_at IS NULL",
        [], |r| r.get(0),
    ).unwrap_or(0);

    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 } else { 0.0 };

    Ok(AccountSummary { today_pnl, trade_count, win_count, win_rate, open_positions })
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 6 — get_settings
// ─────────────────────────────────────────────────────────────────────────────
/// Parses config/settings.py Field defaults, merged with .env overrides.
#[tauri::command]
fn get_settings() -> Result<Value, String> {
    Ok(load_settings_json())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 6b — save_settings  (needed by the Settings section)
// ─────────────────────────────────────────────────────────────────────────────
/// Writes each key in the JSON object to .env as an uppercase env var.
#[tauri::command]
fn save_settings(settings: Value) -> Result<(), String> {
    let map = settings.as_object().ok_or("Expected a JSON object")?;
    for (key, val) in map {
        let env_key = key.to_uppercase();
        let env_val = match val {
            Value::String(s) => s.clone(),
            other            => other.to_string(),
        };
        write_env_var(&env_key, &env_val)?;
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 7 — emergency_halt
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn emergency_halt(state: State<'_, BotProcess>) -> Result<(), String> {
    write_env_var("EMERGENCY_HALT", "true")?;
    let lock = state.child.lock().unwrap();
    if let Some(child) = &*lock {
        let pid = child.id();
        #[cfg(unix)]
        { let _ = Command::new("kill").args(["-USR1", &pid.to_string()]).output(); }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 8 — resume_trading
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn resume_trading() -> Result<(), String> {
    write_env_var("EMERGENCY_HALT", "false")
}

// ─────────────────────────────────────────────────────────────────────────────
// Command 9 — check_python
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn check_python() -> Result<PythonCheck, String> {
    Ok(PythonCheck {
        venv_found: py_exe().exists(),
        env_found:  env_path().exists(),
        db_found:   repo().join("db").exists(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Extra commands used by the UI
// ─────────────────────────────────────────────────────────────────────────────

/// Appends a close request for the symbol to close_queue.txt so that a
/// future Python-side polling loop can honour it.
#[tauri::command]
fn stop_position(symbol: String) -> Result<(), String> {
    let path     = repo().join("close_queue.txt");
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    if !existing.lines().any(|l| l.trim() == symbol.trim()) {
        std::fs::write(&path, format!("{}{}\n", existing, symbol))
            .map_err(|e| format!("Cannot write close_queue.txt: {}", e))?;
    }
    Ok(())
}

/// Aggregated performance data for the Performance section.
#[tauri::command]
fn get_performance(range: String) -> Result<Value, String> {
    let conn = open_db()?;

    let where_clause = match range.as_str() {
        "today" => "WHERE date(opened_at) = date('now','localtime') AND closed_at IS NOT NULL",
        "week"  => "WHERE opened_at >= date('now','localtime','-7 days') AND closed_at IS NOT NULL",
        _       => "WHERE closed_at IS NOT NULL",
    };

    // Summary row
    let (total_pnl, trade_count, win_count): (f64, i64, i64) = conn.query_row(
        &format!("SELECT COALESCE(SUM(gross_pnl),0.0), COUNT(*), \
                  SUM(CASE WHEN gross_pnl > 0 THEN 1 ELSE 0 END) \
                  FROM trades {where_clause}"),
        [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).map_err(|e| e.to_string())?;

    let win_rate = if trade_count > 0 { win_count as f64 / trade_count as f64 } else { 0.0 };

    // Equity curve
    let mut stmt = conn.prepare(&format!(
        "SELECT opened_at, gross_pnl FROM trades {where_clause} ORDER BY opened_at ASC"
    )).map_err(|e| e.to_string())?;

    let mut cumul  = 0f64;
    let mut peak   = 0f64;
    let mut max_dd = 0f64;
    let mut equity: Vec<Value> = Vec::new();

    let rows = stmt.query_map([], |r| Ok((r.get::<_,String>(0)?, r.get::<_,f64>(1)?)))
        .map_err(|e| e.to_string())?;
    for row in rows.filter_map(|r| r.ok()) {
        cumul += row.1;
        if cumul > peak { peak = cumul; }
        let dd = peak - cumul;
        if dd > max_dd { max_dd = dd; }
        equity.push(json!({ "t": row.0, "v": cumul }));
    }

    // Win/loss by signal type
    let mut sstmt = conn.prepare(&format!(
        "SELECT signal_type, \
                SUM(CASE WHEN gross_pnl > 0 THEN 1 ELSE 0 END), \
                SUM(CASE WHEN gross_pnl <= 0 THEN 1 ELSE 0 END) \
         FROM trades {where_clause} GROUP BY signal_type"
    )).map_err(|e| e.to_string())?;

    let by_signal: Vec<Value> = sstmt.query_map([], |r| {
        Ok(json!({
            "signal_type": r.get::<_,Option<String>>(0)?.unwrap_or_default(),
            "wins":        r.get::<_,i64>(1)?,
            "losses":      r.get::<_,i64>(2)?,
        }))
    }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();

    // Simplified Sharpe (mean P&L per trade / std dev, scaled √252)
    let sharpe = if trade_count > 1 {
        let mean = total_pnl / trade_count as f64;
        let mut pstmt = conn.prepare(
            &format!("SELECT gross_pnl FROM trades {where_clause}")
        ).map_err(|e| e.to_string())?;
        let pnls: Vec<f64> = pstmt.query_map([], |r| r.get::<_,f64>(0))
            .map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();
        let variance = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / pnls.len() as f64;
        let std_dev  = variance.sqrt();
        if std_dev > 0.0 { mean / std_dev * 252f64.sqrt() } else { 0.0 }
    } else { 0.0 };

    Ok(json!({
        "total_pnl":    total_pnl,
        "trade_count":  trade_count,
        "win_count":    win_count,
        "win_rate":     win_rate,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "equity_curve": equity,
        "by_signal":    by_signal,
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────────────
fn main() {
    tauri::Builder::default()
        .manage(BotProcess { child: Mutex::new(None) })
        .invoke_handler(tauri::generate_handler![
            start_bot,
            stop_bot,
            get_trades,
            get_open_positions,
            get_account_summary,
            get_settings,
            save_settings,
            emergency_halt,
            resume_trading,
            check_python,
            stop_position,
            get_performance,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Rosetta1");
}
