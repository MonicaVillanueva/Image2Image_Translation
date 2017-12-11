
import argparse




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--parameter1', type=int, default=0)
    parser.add_argument('--parameter2', type=double, default=0)

    config = parser.parse_args()
    print(config)
    main(config)


