//! Quick CLI check of the HuggingFace search helpers.
//! Usage: cargo run --example hfsearch -- "whisper ggml"

fn main() {
    let q = std::env::args().nth(1).unwrap_or_else(|| "whisper ggml".into());
    match dictata::models::parse_hf_query(&q) {
        dictata::models::HfQuery::FileUrl(url, fname) => println!("FileUrl: {url} -> {fname}"),
        dictata::models::HfQuery::Repo(r) => {
            println!("Repo: {r}");
            match dictata::models::hf_list_files(&r) {
                Ok(files) => {
                    for f in files {
                        println!("  {} ({:?} bytes) -> {}", f.fname, f.size, f.url());
                    }
                }
                Err(e) => println!("  ERREUR: {e}"),
            }
        }
        dictata::models::HfQuery::Search(s) => {
            println!("Search: {s}");
            match dictata::models::hf_search(&s) {
                Ok(repos) => {
                    for r in repos {
                        println!("  {r}");
                    }
                }
                Err(e) => println!("  ERREUR: {e}"),
            }
        }
    }
}
