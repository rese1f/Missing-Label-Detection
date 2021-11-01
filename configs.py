import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    
    # parser.add_argument()
    
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args