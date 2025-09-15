import os
import glob

def combine_text_files_from_folder(input_directory, output_file_name="combined_text_data.txt"):
    """
    Reads all text files (.txt) from a specified directory and combines their
    content into a single text file.

    Args:
        input_directory (str): The path to the folder containing the text files.
        output_file_name (str): The name of the output text file.
    """
    # Check if the provided directory path exists.
    if not os.path.isdir(input_directory):
        print(f"Error: The directory '{input_directory}' does not exist.")
        print("Please check the folder path and try again.")
        return

    # Use glob to find all files ending with .txt in the specified directory.
    text_files = glob.glob(os.path.join(input_directory, "*.txt"))

    if not text_files:
        print(f"No text files (.txt) found in the directory: {input_directory}")
        return

    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        print(f"Combining {len(text_files)} text files into '{output_file_name}'...")
        for i, file_path in enumerate(text_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(f"--- Content from {os.path.basename(file_path)} ---/n")
                    outfile.write(content)
                    outfile.write("/n/n")
                    print(f"  ✅ Added content from {os.path.basename(file_path)} ({i+1}/{len(text_files)})")
            except Exception as e:
                print(f"  ❌ Failed to read {os.path.basename(file_path)}: {e}")

    print(f"/n🎉 All text files have been successfully combined into '{output_file_name}'.")

# --- Example Usage ---
if __name__ == "__main__":
    # >>> IMPORTANT: EDIT THE LINE BELOW <<<
    # Replace the placeholder with your actual folder path.
    # A relative path (e.g., "my_text_files") is easiest if the folder is in the same location as this script.
    source_directory = "C:/Users/DELL/Desktop/web_pages (1)/text files" 
    
    # You can change the name of the output text file here.
    output_text_file = "combined_text_data.txt"

    # Call the main function to run the process.
    combine_text_files_from_folder(source_directory, output_text_file)
