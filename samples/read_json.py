import json
annotations = json.load(open( "/home/group/mask_RCNN/datasets/weed/train/via_region_data.json"))
annotations = list(annotations.values())  # don't need the dict keys
annotations = annotations[1]



 # Add images
for key in annotations:
    # Get the x, y coordinaets of points of the polygons that make up
    # the outline of each object instance. These are stores in the
    # shape_attributes (see json format above)
    # The if condition is needed to support VIA versions 1.x and 2.x.
    #print(annotations[key]['regions'])
    polygons = [r['shape_attributes'] for r in annotations[key]['regions']]
    #for r in annotations[key]['regions']:
    #    polygons = r['shape_attributes']
    '''
    if type(a['regions']) is dict:
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
    else:
        polygons = [r['shape_attributes'] for r in a['regions']] 
    '''