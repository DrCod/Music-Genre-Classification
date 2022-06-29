# GCL2022 Project : Music Genre Classification

# To train model:

  python train.py --root_dir path/to/your_generated_spectrograms --model_name densenet201 --family Densenet201 --batch_size 16 --epochs 50 --optimizer_name adamw pretrained 
  
# To predict an audio file:

  python inference.py --root_dir path/to/your_generated_spectrograms --test_path path/to/your_audio_file --model_name densenet201 --family Densenet201 --pretrained_models ./outputs/
  
# To predict a csv file of audio files:

  python inference.py --root_dir path/to/your_generated_spectrograms --test_path path/to/your_audio_csv --model_name densenet201 --family Densenet201 --batch_size 16 --pretrained_models ./outputs/
  
