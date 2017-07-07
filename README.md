# bea12-textquality
Text quality estimation system (BEA12 workshop)

This code implements the method described in
*Transparent text quality assessment with convolutional neural networks*
(to be published in the proceedings of The 12th Workshop on Innovative Use
of NLP for Building Educational Applications, September 8th, Copenhagen).

## Training

The `train_rank.py` program performs training, use the `--help` option to see
all possible arguments.

## Visualization

The `score_rank_visualize.py` script can output color-coded predictions in
HTML or LaTeX.

## Predictions

Due to privacy concerns we are not able to release the full essay dataset used
in our article, but the file `essays.scores` contains our model's predictions
for the essays. The columns are: predicted score, grade 1 (assigned during
blind re-grading), grade 2 (assigned by student's own teacher), length in
characters of essay.

