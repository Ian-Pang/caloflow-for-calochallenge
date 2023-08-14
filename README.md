# CaloFlow for CaloChallenge Dataset 1
## by Claudius Krause, Ian Pang, and David Shih

This repository contains the source code for reproducing the results of

_"CaloFlow for CaloChallenge Dataset 1"_ by Claudius Krause, Ian Pang, and David Shih, [arxiv: 2210.14245](https://arxiv.org/abs/2210.14245)

If you use the code, please cite:
```
@article{Krause:2022jna,
    author = "Krause, Claudius and Pang, Ian and Shih, David",
    title = "{CaloFlow for CaloChallenge Dataset 1}",
    eprint = "2210.14245",
    archivePrefix = "arXiv",
    primaryClass = "physics.ins-det",
    month = "10",
    year = "2022"
}
```

### Running CaloFlow

#### CaloFlow photon teacher (MAF)
To train the CaloFlow photon teacher model, run

`python run_gamma_2023.py --train --data_dir /path/to/data_directory --hidden_size 378 --output_dir /path/to/output_directory  --n_epochs 100 --with_noise`

To generate 100k events from the CaloFlow photon teacher model, run

`python run_gamma_2023.py --generate_to_file --restore_file /path/to/teacher_weights --hidden_size 378 --output_dir /path/to/output_directory`

#### CaloFlow photon student (IAF)

To train the CaloFlow photon student model, run

`python run_gamma_2023.py --train --student_mode --restore_file /path/to/teacher_weights  --fully_guided --batch_size 175 --n_epochs 100 --hidden_size 378 --student_hidden_size 736 --output_dir /path/to/output_directory --with_noise --data_dir /path/to/data_directory`

To generate 100k events from the CaloFlow photon student model, run

`python run_gamma_2023.py --generate_to_file --student_mode --student_restore_file /path/to/student_weights --student_hidden_size 736 --output_dir /path/to/output_directory --fully_guided`

#### CaloFlow pion teacher (MAF)

To train the CaloFlow pion teacher model, run

`python run_piplus_2023.py --train --data_dir /path/to/data_directory --hidden_size 533  --output_dir /path/to/output_directory --with_noise --batch_size 500 --n_epochs 100`

To generate 100k events from the CaloFlow pion teacher model, run

`python run_piplus_2023.py --generate_to_file --restore_file /path/to/teacher_weights  --hidden_size 533  --output_dir /path/to/output_directory`

#### CaloFlow pion student (IAF)

To train the CaloFlow pion student model, run

`python run_piplus_2023.py --train --student_mode --restore_file /path/to/teacher_weights  --hidden_size 533 --student_hidden_size 500 --batch_size 175 --output_dir /path/to/output_directory --with_noise --n_epochs 150 --data_dir /path/to/data_directory`

To generate 100k events from the CaloFlow pion student model, run

`python run_piplus_2023.py --generate_to_file --student_mode --student_restore_file /path/to/student_weights --student_hidden_size 500 --output_dir /path/to/output_directory` 