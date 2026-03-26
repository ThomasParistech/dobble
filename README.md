# Generate and print you custom Dobble Game

This repository contains the implementation of a generator for the Dobble game (also known as Spot It!), a popular card game known for its fun and fast-paced gameplay. The game is based on a set of cards, where each card has a unique combination of symbols, but every pair of cards shares exactly one symbol in common.

This generator is designed to let you provide 57 custom symbols, and it will automatically generate a print-ready PDF containing 57 cards, each featuring 8 symbols.

Using the junior version, you can provide 31 custom symbols, and it will automatically generate 31 cards, each featuring 6 symbols.

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
python3 -m dobble --symbols_folder assets/symbols_examples/ --output_folder data/result
```

To run the junior version, add the `junior_size` flag:
```
python3 -m dobble --junior_size --symbols_folder assets/symbols_examples_junior/ --output_folder data/result
```

To use cards with hexagonal shapes (easier to cut), add the `hexagon` flag:
```
python3 -m dobble --hexagon --symbols_folder assets/symbols_examples/ --output_folder data/result
```

To generate a PDF with card backs for **recto-verso (double-sided) printing**, use the `back_image_path` option:
```
python3 -m dobble --back_image_path assets/back.png --symbols_folder assets/symbols_examples/ --output_folder data/result
```
This inserts a back page after each front page in the PDF. The back image is resized to fill the card bounding box (works with both circular and hexagonal cards) and the page layout is horizontally mirrored so that card backs align with their fronts when printed double-sided (long-edge flip). Use your printer's duplex/recto-verso setting to print.

*Tip: you can use a "circle punch" to cut the printed cards.*


# Example
![](./doc/dobble_evolution.gif)



# Contributing

Contributions are welcome!

# License

This project is licensed under the non-commercial [CC BY-NC-SA 4.0 License](LICENSE).


