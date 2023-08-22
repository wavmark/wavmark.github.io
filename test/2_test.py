import subprocess
import glob


def run_cmd(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print("Error executing the command:", e)
        output = e.output

    return output


def test_decode_host():
    for f in all_audio_files:
        print("f:", f)
        command = ["python", "1_wavmark.py", "--mode=decode", "--input=" + f]
        output = run_cmd(command)
        assert "No Watermark Found" in output


def test_encode_decode():
    for f in all_audio_files:
        print(f)
        f_name = f.split("/")[-1]
        f_name_wav = f_name.replace(".mp3", ".wav")
        tmp_output = "/tmp/" + f_name_wav

        command1 = ["python", "1_wavmark.py", "--input=" + f, "--output=" + tmp_output, "--watermark=0010101011000111"]
        print(" ".join(command1))
        output = run_cmd(command1)
        print(output)

        command2 = ["python", "1_wavmark.py", "--mode=decode", "--input=" + tmp_output]
        output = run_cmd(command2)
        print(" ".join(command2))

        assert "0010101011000111" in output, output


if __name__ == "__main__":
    folder_path = "../data"
    wav_files = glob.glob(folder_path + "/*.wav")
    mp3_files = glob.glob(folder_path + "/*.mp3")
    all_audio_files = wav_files + mp3_files
    # all_audio_files = mp3_files

    test_decode_host()
    test_encode_decode()
