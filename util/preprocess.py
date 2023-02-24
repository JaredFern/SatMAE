import pandas as pd


def add_img_name(split, split_csv):
    """ fmow-sentinel/<split>/<category>/<category>_<location_id>/<category>_<location_id>_<image_id>.tif """
    split_meta = pd.read_csv(split_csv, index_col=0)
    
    categories = split_meta['category'].astype(str)
    location_id = split_meta['location_id'].astype(str)
    image_id = split_meta['image_id'].astype(str)
    
    img_dirs = f"fmow-sentinel/{split}/" + categories + "/" + categories + "_" + location_id + "/"  
    img_name =  categories + "_" + location_id + "_" + image_id + ".tif"
    split_meta['img_name'] = img_dirs + img_name
    
    split_meta.to_csv(f"{split}.csv")
    