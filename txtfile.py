import os

def append_text_to_file(output_file_path, text_to_append, question_to_append, max_size_kb):
    if os.path.exists(output_file_path):
        current_size_kb = os.path.getsize(output_file_path) / 1024
        if current_size_kb > max_size_kb:
            base, ext = os.path.splitext(output_file_path)
            new_file_index = 1
            while os.path.exists(f"{base}_{new_file_index}{ext}"):
                new_file_index += 1
            output_file_path = f"{base}_{new_file_index}{ext}"

    with open(output_file_path, 'a') as output_file:
        output_file.write("Question:")
        output_file.write(question_to_append)
        output_file.write('\n')
        output_file.write("Corrected Answer:")
        output_file.write(text_to_append)
        output_file.write('\n')
