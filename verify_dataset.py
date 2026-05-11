import os
from pathlib import Path

dataset_path = Path('images/train')
print('📊 Dataset Extraction Verification')
print('=' * 55)

total = len(list(dataset_path.rglob("*.jpg")))
print(f'✅ Total images: {total}\n')

print('Per emotion category:')
print('-' * 55)
for emotion_dir in sorted(dataset_path.iterdir()):
    if emotion_dir.is_dir():
        count = len(list(emotion_dir.glob('*.jpg')))
        print(f'  {emotion_dir.name:<12} {count:>6} images')

print('=' * 55)
print('\n✅ Dataset extraction COMPLETE!')
print('\nNext step: Run evaluation')
print('  python evaluation/evaluate_model.py --data ./images/train')
