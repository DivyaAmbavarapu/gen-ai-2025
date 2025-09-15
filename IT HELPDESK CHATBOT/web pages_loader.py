import os
import glob

def combine_web_pages_to_txt_from_folder(input_directory, output_file_name="combined_web_data.txt"):
    """
    Reads all common web page files (.html, .htm) from a specified directory
    and combines their content into a single text file.

    Args:
        input_directory (str): The path to the folder containing the web page files.
        output_file_name (str): The name of the output text file.
    """
    # Check if the provided directory path exists.
    if not os.path.isdir(input_directory):
        print(f"Error: The directory '{input_directory}' does not exist.")
        print("Please check the folder path and try again.")
        return

    # Use glob to find all files with common web page extensions.
    web_page_files = []
    for extension in ['*.html', '*.htm']:
        web_page_files.extend(glob.glob(os.path.join(input_directory, extension)))

    if not web_page_files:
        print(f"No web page files found in the directory: {input_directory}")
        return

    with open(output_file_name, 'w', encoding='utf-8') as outfile:
        print(f"Combining {len(web_page_files)} web page files into '{output_file_name}'...")
        for i, file_path in enumerate(web_page_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(f"--- Content from {os.path.basename(file_path)} ---/n")
                    outfile.write(content)
                    outfile.write("/n/n")
                    print(f"  ✅ Added content from {os.path.basename(file_path)} ({i+1}/{len(web_page_files)})")
            except Exception as e:
                print(f"  ❌ Failed to read {os.path.basename(file_path)}: {e}")

    print(f"/n🎉 All web page files have been successfully combined into '{output_file_name}'.")

# --- Example Usage ---
if __name__ == "__main__":
    # >>> IMPORTANT: EDIT THE LINE BELOW <<<
    # Replace the placeholder with your actual folder path.
    # A relative path (e.g., "my_web_pages") is easiest if the folder is in the same location as this script.
    source_directory = "C:/Users/DELL/Desktop/web_pages (1)/web_pages" 
    
    # You can change the name of the output text file here.
    output_text_file = "combined_web_data.txt"

    # Call the main function to run the process.
    combine_web_pages_to_txt_from_folder(source_directory, output_text_file)
