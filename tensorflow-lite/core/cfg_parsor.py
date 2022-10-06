# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:36:14 2022

@author: user
"""

def model_parse(cfg):
    result = []
    
    cfg = cfg.split("\n")
    
    layer_flag = -1
    layer_flag_dict = { "convolutional": [0, 
                        { "name": "convolutional",
                          "batch_normalize": '0',
                          "filters": '1',
                          "size": '1',
                          "stride": '1',
                          "pad": '1',
                          "activation": 'linear' }],
                       
                        "route": [1,
                         { "name": "route",
                           "groups": '0',
                           "group_id": '0',
                           "layers": '-1' }],
                        
                        "maxpool": [2, 
                        { "name": "maxpool",
                          "size": '2',
                          "stride": '2' }],
                        
                        "upsample": [3,
                        { "name": "upsample",
                          "stride": '1' }]
                      }
    record = {}
    
    for s in cfg:
        s = s.lower().strip()
        
        if s == "" or s.startswith("#"): continue
        
        if s.startswith("[") and s.endswith("]"):
            layer = s[1:-1].strip()
            result.append(record)
            
            if layer not in layer_flag_dict.keys():
                record = { "name": layer }
                layer_flag = -1
            else:
                layer_flag, default_value = layer_flag_dict[layer]
                record = default_value.copy()
        
        else:
            r_key, r_value = s.split("=", maxsplit=1)
            
            r_key = r_key.strip()
            r_value = r_value.strip()
            record[r_key] = r_value
            
    result.append(record)
    
    temp = result
    result = []
    
    for t in temp:
        if t:
            result.append(t)
    
    return result[1:]