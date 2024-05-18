use serde_json::{json, Value};
use time::OffsetDateTime;
use std::env;
use std::io::{BufReader, BufRead};
use std::fs::File;
use qdrant::*;
use wasmedge_wasi_nn::{
    self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};
use clap::{Arg, ArgMatches, Command};

async fn generate_upsert (context: &mut GraphExecutionContext, data: &str, client: &qdrant::Qdrant, id: u64, collection_name: &str, vector_size: usize, start_vector_id: u64) {
    set_data_to_context(context, data.as_bytes().to_vec()).unwrap();
    match context.compute() {
        Ok(_) => (),
        Err(Error::BackendError(BackendError::ContextFull)) => {
            println!("\n[INFO] Context full");
        }
        Err(Error::BackendError(BackendError::PromptTooLong)) => {
            println!("\n[INFO] Prompt too long");
        }
        Err(err) => {
            println!("\n[ERROR] {}", err);
        }
    }
    let embd = get_embd_from_context(&context, vector_size);

    let mut embd_vec = Vec::<f32>::new();
    for idx in 0..vector_size as usize {
        embd_vec.push(embd["embedding"][idx].as_f64().unwrap() as f32);
    }

    println!("{} : ID={} Size={} Points ID={}", OffsetDateTime::now_utc(), id, embd_vec.len(), start_vector_id + id);

    let mut points = Vec::<Point>::new();
    points.push(Point{
        id: PointId::Num(start_vector_id + id), 
        vector: embd_vec,
        payload: json!({"source": data}).as_object().map(|m| m.to_owned()),
    });

    // Upsert each point (you can also batch points for upsert)
    let r = client.upsert_points(collection_name, points).await;
    println!("Upsert points result is {:?}", r);
}

fn set_data_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), Error> {
    context.set_input(0, TensorType::U8, &[1], &data)
}

#[allow(dead_code)]
fn set_metadata_to_context(
    context: &mut GraphExecutionContext,
    data: Vec<u8>,
) -> Result<(), Error> {
    context.set_input(1, TensorType::U8, &[1], &data)
}

fn get_data_from_context(context: &GraphExecutionContext, vector_size: usize, index: usize) -> String {
    // Preserve for tokens with average token length 15
    let max_output_buffer_size: usize = vector_size * 15 + 128;
    let mut output_buffer = vec![0u8; max_output_buffer_size];
    let mut output_size = context.get_output(index, &mut output_buffer).unwrap();
    output_size = std::cmp::min(max_output_buffer_size, output_size);

    String::from_utf8_lossy(&output_buffer[..output_size]).to_string()
}

fn get_embd_from_context(context: &GraphExecutionContext, vector_size: usize) -> Value {
    let embd = &get_data_from_context(context, vector_size, 0);
    // println!("\n[EMBED] {}", embd);
    serde_json::from_str(embd).unwrap()
}
fn check_level(text: &str) -> u32 {
    let mut level = 0;
    if text.starts_with("#") {
        for (_, c) in text.char_indices() {
            if c == '#' {
                level += 1;
            } else {
                break;
            }
        }
        if level <= 6 && text.chars().nth(level).unwrap() == ' ' {
            return level as u32;
        } 
    }
    0
}

fn parse_parameter(args: &Vec<String>) -> ArgMatches {
    let matches = Command::new("markdown_embed")
        .version("1.0")
        .about("Create embeddings from a markdown docs")
        .disable_help_subcommand(true)
        .arg(
            Arg::new("maximum_context_length")
                .long("maximum_context_length")
                .short('m')
                .value_name("maximum_context_length")
                .value_parser(clap::value_parser!(usize))
                .help("Maximum context length limitation. If exceeds it, the context will be truncated.")
        )
        .arg(
            Arg::new("start_vector_id")
                .long("start_vector_id")
                .short('s')
                .value_name("start_vector_id")
                .default_value("0")
                .value_parser(clap::value_parser!(u64))
                .help("Start vector id. It defaults to 0"),
        )
        .arg(
            Arg::new("heading_level")
            .long("heading_level")
            .short('l')
            .value_name("heading_level")
            .default_value("1")
            .help("Mardown heading level for generate chunks.")
            .value_parser(clap::value_parser!(u32)),
        )
        .arg(
            Arg::new("ctx_size")
            .long("ctx_size")
            .short('c')
            .value_name("ctx_size")
            .default_value("512")
            .help("Context size. It defaults to 512.")
            .value_parser(clap::value_parser!(usize)),
        )
        .get_matches_from(args.clone().split_off(4));
    return matches;
}
#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();
    let model_name: &str = &args[1];
    let collection_name: &str = &args[2];
    let vector_size: usize = args[3].trim().parse().unwrap();
    let file_name: &str = &args[4];

    let matches = parse_parameter(&args);
    let ctx_size: usize = *matches.get_one("ctx_size").unwrap();

    let mut options = json!({});
    options["embedding"] = serde_json::Value::Bool(true);
    options["ctx-size"] = serde_json::Value::from(ctx_size);
    options["batch-size"] = serde_json::Value::from(ctx_size);

    let graph =
        GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
            .config(options.to_string())
            .build_from_cache(model_name)
            .expect("Create GraphBuilder Failed, please check the model name or options");
    let mut context = graph
        .init_execution_context()
        .expect("Init Context Failed, please check the model");

    let client = qdrant::Qdrant::new();

    let start_vector_id = *matches.get_one("start_vector_id").unwrap();
    let heading_level = *matches.get_one("heading_level").unwrap();
    let mut id : u64 = 0;
    let mut current_section = String::new();
    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut code_mode = false;
    let mut prefix: Vec<String>  = vec!();
    for line_result in reader.lines() {
        let line = line_result?;
        if line.trim().starts_with("```") {
            code_mode = !code_mode;
        }
        let level = check_level(&line);
        let section_heading = check_level(&current_section);
        if !current_section.trim().is_empty() && level != 0
            && (level == heading_level || level < heading_level && section_heading == heading_level ){
            let mut prefix_data:String = String::new();
            for head in prefix.iter() {
                if check_level(&head) >= section_heading{
                    break;
                }
                prefix_data += head;
                prefix_data += "\n";
            }
            current_section = prefix_data + &current_section;
            if let Some(&maximum) = matches.get_one::<usize>("maximum_context_length") {
                if current_section.len() > maximum {
                    println!("\n [WARNING] Index: {} exceed maximum contex length limitation.", id);
                    current_section = current_section.chars().take(maximum).collect();
                }
            }
            generate_upsert(&mut context, &current_section, &client, id, collection_name, vector_size, start_vector_id).await;
            id += 1;
            // Start a new section
            current_section.clear();
        }
        if level < heading_level && level != 0 {
            while let Some(last) = prefix.last() {
                if check_level(last) >= level {
                    prefix.pop();
                }else {
                    break;
                }   
            }
            prefix.push(line.clone());
        }
        current_section.push_str(&line);
        current_section.push('\n');
    }

    // The last segment
    if !current_section.trim().is_empty() {
        let section_heading = check_level(&current_section);
        let mut prefix_data:String = String::new();
        for head in prefix.iter() {
            if check_level(&head) >= section_heading{
                break;
            }
            prefix_data += head;
            prefix_data += "\n";
        }
        current_section = prefix_data + &current_section;
        if let Some(&maximum) = matches.get_one::<usize>("maximum_context_length") {
            if current_section.len() > maximum {
                println!("\n [WARNING] Index: {} exceed maximum contex length limitation.", id);
                current_section = current_section.chars().take(maximum).collect();
            }
        }
        generate_upsert(&mut context, &current_section, &client, id, collection_name, vector_size, start_vector_id).await;
    }
    Ok(())
}
