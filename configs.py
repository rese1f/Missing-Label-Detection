import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    
    parser.add_argument('--dataset', default='', type=str,
                        help='path to dataset')
    parser.add_argument('--checkpoint', default='',type=str,
                        help='name of checkpoint model weight')
    
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args