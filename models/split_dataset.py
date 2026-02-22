import splitfolders

print('Starting dataset split...')
splitfolders.ratio(r"C:\Users\almog\Desktop\Studies\תואר\שנה ג\Deep Learning\Project Dataset\characters_dataset",
                   output="output_dataset",
                   seed=1337, ratio=(.8, .2), group_prefix=None, move=False)

print("Data split complete! Check the 'output_dataset' folder.")
