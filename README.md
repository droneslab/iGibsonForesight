## To use:
1. Clone repository
2. Download the folowing:
   * Assets: https://storage.googleapis.com/gibson_scenes/assets_igibson.tar.gz -- extract to igibson/data/assets
   * ig_dataset: https://storage.googleapis.com/gibson_scenes/ig_dataset.tar.gz -- extract to igibson/data/ig_dataset
   * episodes_data: https://mega.nz/folder/cFh0DTYR#tLNZ8IdtUXoHIdMpLQCKlQ -- extract to igibson/data/episodes_data
3. `chmod -R 777 igibson`
4. To ensure that everything is set up correctly, run `test.py` for 1-2 minutes until you see "Rollout Time: X.XXXX seconds" being printed.
5. Use `full_train.sh` to run experiments.
___

###  Users might need to create a top-level `/models` and/or `/logs` directory before training.
### Models will not be uploaded to GitHub due to file size, and therefore should be shared through some other method.ed to GitHub due to file size, and therefore should be shared through some  other method.