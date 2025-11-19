import os
import csv

def save_results(save_path, iteration, total, correct_audio=None,
                 correct_image=None, correct_audio_visual=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Define columns and values dynamically
    columns = ["iteration", "audio_text_accuracy", "image_text_accuracy", "av_accuracy"]
    values = [iteration]

    # Helper for converting values
    def fmt(val):
        if val is None:
            return "-1"
        return f"{val / total:.4f}"

    values.append(fmt(correct_audio))
    values.append(fmt(correct_image))
    values.append(fmt(correct_audio_visual))

    # Check if file exists
    file_exists = os.path.exists(save_path)

    with open(save_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(columns)

        writer.writerow(values)
