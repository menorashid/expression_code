import sys;
import visualize;

def main(args):
	# folder=args[1];
	visualize.writeHTMLForFolder(args[1],args[2]);

if __name__=='__main__':
	main(sys.argv);