import sys
import os
import time

def process_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    results = []
    results.append(f"File: {os.path.basename(file_path)}")
    results.append(f"Text: {content}")
    return "\n".join(results)

def remove_empty_lines(text):
    return "\n".join(line for line in text.splitlines() if line.strip())

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 paras_from_folder.py input_folder output.txt")
        sys.exit(1)

    input_folder, output_file = sys.argv[1], sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory")
        sys.exit(1)

    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    total_files = len(txt_files)

    print(f"Total files to process: {total_files}")

    start_time = time.time()
    with open(output_file, 'w') as f:
        for i, txt_file in enumerate(txt_files):
            file_start_time = time.time()
            
            file_path = os.path.join(input_folder, txt_file)
            processed_file = process_file(file_path)
            processed_file = remove_empty_lines(processed_file)
            f.write(processed_file + "\n\n")

            file_end_time = time.time()
            file_duration = file_end_time - file_start_time

            if i == 0:
                estimated_total_time = file_duration * total_files
                print(f"Estimated total time: {format_time(estimated_total_time)}")

            # Update progress
            files_left = total_files - (i + 1)
            estimated_time_left = file_duration * files_left
            progress = (i + 1) / total_files * 100

            print(f"\rProgress: {progress:.2f}% | Estimated time remaining: {format_time(estimated_time_left)}", end="")

    end_time = time.time()
    total_duration = end_time - start_time

    print(f"\nProcessing complete. Total time taken: {format_time(total_duration)}")
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()
