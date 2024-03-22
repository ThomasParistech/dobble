[Link to the Medium article](https://medium.com/better-programming/generate-and-print-your-custom-dobble-274dc888a33e)

Run
```
python3 -m dobble --symbols_folder <SYMBOL_FOLDER> --output_folder <RESULT_FOLDER>
```
to execute the 4 following steps:
- Make all images square and rotation-proof
- Extract the binary mask of the symbols images
- Generate 57 cards with randomly drawn symbols
- Merge Dobble cards into a scaled PDF ready to print

To try on toy data, run
```
python3 -m dobble --symbols_folder images/symbols_examples/ --output_folder result
```

![](./images/dobble_evolution.gif)
