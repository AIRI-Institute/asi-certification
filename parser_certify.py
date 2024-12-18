import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Certify many examples')

    parser.add_argument("--num_query_val",
                        type=int,
                        default=5,
                        help="number of query samples")
    
    parser.add_argument("--classes_per_it_tr",
                        type=int,
                        default=5,
                        help="number of classes per iteration")
    
    parser.add_argument("--classes_per_it_val",
                        type=int,
                        default=5,
                        help="number of classes per iteration")
    
    parser.add_argument("--iterations",
                        type=int,
                        default=100,
                        help="iterations per epoch")
    
    parser.add_argument("--skip",
                        type=int,
                        default=1,
                        help="how many examples to skip")
    
    parser.add_argument("--max",
                        type=int,
                        default=-1,
                        help="stop after this many examples")   
    
    parser.add_argument("--N",
                        type=int,
                        default=1000,
                        help="number of samples to use for smoothing")
    
    parser.add_argument("--M",
                        type=int,
                        default=50,
                        help="number of samples to use for confidence intervals")
    
    parser.add_argument("--K",
                        type=int,
                        default=5,
                        help="maximum repeats number to increase number of samples")
    
    parser.add_argument("--batch",
                        type=int,
                        default=100,
                        help="batch size")   
    
    parser.add_argument("--alpha",
                        type=float,
                        default=0.001,
                        help="failure probability")

    parser.add_argument('-imsize', '--orig_imsize',
                        type=int,
                        default=-1,
                        help='-1 for no cache, and -2 for no resize, only for MiniImageNet')
        
    parser.add_argument("--cuda",
                        default=True)
    
    parser.add_argument("--cuda_number",
                        type=int,
                        default=3)
    
    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        default=7,
                        help='input for the manual seeds initializations')
    
    parser.add_argument("--mode",
                        type=str,
                        default='small-norm')
    
    parser.add_argument("--descr",
                        type=str,
                        default=None,
                        help="additional description for output filename")
    
    return parser
