use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = Path::new(&out_dir).join("words_generated.rs");

    let words_content = fs::read_to_string("words.txt").unwrap_or_default();
    let mut words: BTreeSet<&str> = BTreeSet::new();

    for line in words_content.lines() {
        let word = line.trim();
        if word.len() >= 2 && word.len() <= 15 && word.chars().all(|c| c.is_ascii_alphabetic()) {
            words.insert(word);
        }
    }

    let mut code = String::from("pub static DICTIONARY_WORDS: &[&str] = &[\n");
    for word in &words {
        code.push_str(&format!("    \"{}\",\n", word));
    }
    code.push_str("];\n");

    fs::write(dest_path, code).unwrap();
    println!("cargo:rerun-if-changed=words.txt");
}
