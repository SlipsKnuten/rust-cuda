mod extract_llfns;
mod ptx_cache;

use pico_args::Arguments;
use std::{error::Error, path::Path};

use crate::extract_llfns::extract_llfns;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = Arguments::from_env();
    let sub = args.subcommand()?.unwrap_or_default();

    match sub.as_str() {
        "extract_llfns" => {
            let arg1 = args.free_from_str::<String>()?;
            let file = Path::new(&arg1);
            let arg2 = args.free_from_str::<String>()?;
            let dir = Path::new(&arg2);
            args.finish();
            extract_llfns(file, dir);
            Ok(())
        }
        "cache" => {
            let cache_cmd = args.free_from_str::<String>()?;
            args.finish();
            match cache_cmd.as_str() {
                "stats" => ptx_cache::stats()?,
                "clear" => ptx_cache::clear()?,
                "enable" => ptx_cache::enable(),
                "disable" => ptx_cache::disable(),
                _ => ptx_cache::usage(),
            }
            Ok(())
        }
        _ => {
            eprintln!("Unknown command, available commands:");
            eprintln!("  extract_llfns");
            eprintln!("  cache [stats|clear|enable|disable]");
            std::process::exit(1);
        }
    }
}
