fn 流式输出(text: &str) -> io::Result<()> {
    for c in text.chars() {
        print!("{}", c);
        io::stdout().flush()?;
        thread::sleep(Duration::from_millis(100));
    }
    println!();
    Ok(())
}
