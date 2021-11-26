import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    
    parser.add_argument('--dataset', default='/mnt/sdb/MLD', type=str,
                        help='path to dataset')
    parser.add_argument('--checkpoint', default='',type=str,
                        help='name of checkpoint model weight')
    parser.add_argument('--batch-size', default=36, type=int,
                        help='training batch size')
    parser.add_argument('--epoch', default=10, type=int,
                        help='training epoch')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--size', default=640, type=int,
                        help='image size')
    
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args