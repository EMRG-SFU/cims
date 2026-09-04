"""Universal dictionary operations - work with any data"""
import numpy as np


def sum_dicts(*dicts):
    """Sum values from multiple dictionaries"""
    out = {}
    all_keys = set()
    for d in dicts:
        all_keys |= set(d.keys())
    
    for k in all_keys:
        total = 0
        has_valid = False
        for d in dicts:
            v = d.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                total += float(v)
                has_valid = True
        out[k] = total if has_valid else None
    return out


def weighted_average_dicts(data_dicts, weight_dicts):
    """Calculate weighted average of multiple dictionaries"""
    out = {}
    all_keys = set()
    for d in data_dicts:
        all_keys |= set(d.keys())
    
    for k in all_keys:
        weighted_sum = 0
        total_weight = 0
        has_valid = False
        
        for data_dict, weight_dict in zip(data_dicts, weight_dicts):
            data_val = data_dict.get(k)
            weight_val = weight_dict.get(k)
            
            if (data_val is not None and not (isinstance(data_val, float) and np.isnan(data_val)) and
                weight_val is not None and not (isinstance(weight_val, float) and np.isnan(weight_val))):
                weighted_sum += float(data_val) * float(weight_val)
                total_weight += float(weight_val)
                has_valid = True
        
        out[k] = (weighted_sum / total_weight) if has_valid and total_weight > 0 else None
    return out


def divide_dicts(numerator: dict, denominator: dict):
    """Divide values in numerator by values in denominator"""
    out = {}
    for k in numerator.keys():
        num = numerator.get(k)
        denom = denominator.get(k)
        if num is not None and not (isinstance(num, float) and np.isnan(num)) and \
           denom is not None and not (isinstance(denom, float) and np.isnan(denom)) and denom != 0:
            out[k] = float(num) / float(denom)
        else:
            out[k] = None
    return out

def multiply_dicts(dict1: dict, dict2: dict):
    """Mutiply values in dict 1 by values in dict2"""
    out = {}
    for k in dict1.keys():
        val1 = dict1.get(k)
        val2 = dict2.get(k)
        if val1 is not None and not (isinstance(val1, float) and np.isnan(val1)) and \
            val2 is not None and not (isinstance(val2, float) and np.isnan(val2)):
            out[k] = float(val1) * float(val2)
        else:
            out[k] = None
    return out

