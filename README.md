# Medium Article

For a detailed explanation of the algorithm used, please refer to my [`Medium article`](https://medium.com/better-programming/generate-and-print-your-custom-dobble-274dc888a33e) where I explain the different steps of the algorithm.



# Google Colab Notebook

If you'd like to avoid the hassle of setting up your Python environment, you can effortlessly use this dedicated [`Google Colab notebook`](./colab_notebook.ipynb).

Simply run it on [`Google Colab`](https://colab.research.google.com/).



# Local Run

Run
```
python3 -m dobble --symbols_folder <SYMBOL_FOLDER> --output_folder <RESULT_FOLDER>
```
to execute the 3 following steps:
- Make all images square and rotation-proof and extract their binary masks
- Generate 57 cards with randomly drawn symbols
- Merge Dobble cards into a scaled PDF ready to print

To try on toy data, run
```
python3 -m dobble --symbols_folder images/symbols_examples/ --output_folder result
```


# Example
![](./images/dobble_evolution.gif)


