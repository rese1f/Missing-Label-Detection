import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Experience script')
    parser.add_argument('--name', default='test', type=str,
                        help='name for test')
    parser.add_argument('--dataset', default='/mnt/sdb/MLD', type=str,
                        help='path to dataset')
    parser.add_argument('--checkpoint', default='epoch_50.pth',type=str,
                        help='name of checkpoint model weight')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='training batch size')
    parser.add_argument('--epoch', default=300, type=int,
                        help='training epoch')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate')
    parser.add_argument('--size', default=640, type=int,
                        help='image size')
    parser.add_argument('--val', default=False, action='store_true',
                        help="vaildate while training")
    
    args = parser.parse_args()
    
    # if configs conflict:
    #   raise Keyerror()
    
    return args