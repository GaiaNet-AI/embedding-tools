use serde_json::{json, Value};
use time::OffsetDateTime;
use std::env;
use std::process;
use std::io::BufReader;
use std::fs::File;
use qdrant::*;
use wasmedge_wasi_nn::{
    self, BackendError, Error, ExecutionTarget, GraphBuilder, GraphEncoding, GraphExecutionContext,
    TensorType,
};
use clap::{Arg, ArgMatches, Command};
use uuid::Uuid;

async fn generate_upsert (context: &mut GraphExecutionContext, full: &str, summary: &str, client: &qdrant::Qdrant, uuid: &str, collection_name: &str, vector_size: usize) -> Result<(), Error> {
    set_data_to_context(context, summary.as_bytes().to_vec()).unwrap();
    match context.compute() {
        Ok(_) => (),
        Err(Error::BackendError(BackendError::ContextFull)) => {
            println!("\n[INFO] Context full");
            return Err(Error::BackendError(BackendError::ContextFull));
        }
        Err(Error::BackendError(BackendError::PromptTooLong)) => {
            println!("\n[INFO] Prompt too long");
            return Err(Error::BackendError(BackendError::PromptTooLong));
        }
        Err(err) => {
            println!("\n[ERROR] {}", err);
            return Err(err);
        }
    }
    let embd = get_embd_from_context(&context, vector_size);

    let mut embd_vec = Vec::<f32>::new();
    for idx in 0..vector_size as usize {
        embd_vec.push(embd["embedding"][idx].as_f64().unwrap() as f32);
    }

    println!("{} : Size={} Points ID={}", OffsetDateTime::now_utc(), embd_vec.len(), uuid);

    let mut points = Vec::<Point>::new();
    points.push(Point{
        id: PointId::Uuid(uuid.to_string()), 
        vector: embd_vec,
        payload: json!({"source": full, "search": summary}).as_object().map(|m| m.to_owned()),
    });

    // Upsert each point (you can also batch points for upsert)
    let r = client.upsert_points(collection_name, points).await;
    println!("Upsert points result is {:?}", r);
    Ok(())
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

fn parse_parameter(args: &Vec<String>) -> ArgMatches {
    let matches = Command::new("csv_embed")
        .version("1.0")
        .about("Create embeddings from a CSV docs")
        .disable_help_subcommand(true)
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
    options["ubatch-size"] = serde_json::Value::from(ctx_size);

    let graph =
        GraphBuilder::new(GraphEncoding::Ggml, ExecutionTarget::AUTO)
            .config(options.to_string())
            .build_from_cache(model_name)
            .expect("Create GraphBuilder Failed, please check the model name or options");
    let mut context = graph
        .init_execution_context()
        .expect("Init Context Failed, please check the model");

    let client = qdrant::Qdrant::new();

    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut rdr = csv::Reader::from_reader(reader);
    for result in rdr.records() {
        match result {
            Ok(record) => {
                let full = &record[0];
                let summary = &record[1];
                let uuid = Uuid::new_v4();
                match generate_upsert(&mut context, &full, &summary, &client, &uuid.to_string(), collection_name, vector_size).await {
                    Ok(_) => (),
                    Err(_err) => {
                        println!("\n FAILED summary: {}", &summary);
                    }
                }
            },
            Err(err) => {
                println!("error reading CSV: {}", err);
                process::exit(1);
            }
        }
    }
    Ok(())
}
