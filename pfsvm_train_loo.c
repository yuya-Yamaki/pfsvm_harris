#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <limits.h>
#include "svm.h"
#include "pfsvm.h"
struct svm_parameter param;
struct svm_problem prob;
struct svm_model *model;
struct svm_node *x_space;
/*******************Harris******************/
// struct svm_parameter param_harris;
// struct svm_problem prob_harris;
// struct svm_model *model_harris;
// struct svm_node *x_space_harris;
/*******************************************/
#define LEAVE_ONE_OUT
#define RND_SEED 12345L
#ifdef LEAVE_ONE_OUT
#define MAX_IMAGE 256
#define SAMPLE_RATIO 0.01
#else
#define MAX_IMAGE 1
#define SAMPLE_RATIO 1.01
#endif

int set_images(char *org_dir, char *dec_dir, IMAGE **oimg_list, IMAGE **dimg_list)
{
#ifdef LEAVE_ONE_OUT
	FILE *fp;
	DIR *dir;
	struct dirent *dp;
	char org_img[256], dec_img[256];
	int num_img = 0;

	if ((dir = opendir(org_dir)) == NULL)
	{
		fprintf(stderr, "Can't open directory '%s'\n", org_dir);
		exit(1);
	}
	while ((dp = readdir(dir)) != NULL)
	{
		if (strncmp(dp->d_name + strlen(dp->d_name) - 4, ".pgm", 4) != 0)
			continue;
		strncpy(org_img, org_dir, 255);
		if (org_img[strlen(org_img) - 1] != '/')
		{
			strcat(org_img, "/");
		}
		strcat(org_img, dp->d_name);
		strncpy(dec_img, dec_dir, 255);
		if (dec_img[strlen(dec_img) - 1] != '/')
		{
			strcat(dec_img, "/");
		}
		strcat(dec_img, dp->d_name);
		strcpy(dec_img + strlen(dec_img) - 4, "-dec.pgm");
		if ((fp = fopen(dec_img, "r")) == NULL)
			continue;
		fclose(fp);
		printf("%s %s\n", org_img, dec_img);
		oimg_list[num_img] = read_pgm(org_img);
		dimg_list[num_img] = read_pgm(dec_img);
		num_img++;
	}
	return (num_img);
#else
	oimg_list[0] = read_pgm(org_dir);
	dimg_list[0] = read_pgm(dec_dir);
	return (1);
#endif
}

int main(int argc, char **argv)
{
	IMAGE *oimg_list[MAX_IMAGE], *dimg_list[MAX_IMAGE], *org, *dec;
	int cls[MAX_CLASS];
	int i, j, k, m, n, label, img;
	int num_img, num_class = 3;
	size_t elements;
	double th_list[MAX_CLASS / 2], fvector[NUM_FEATURES], sig_gain = 1.0;
	const char *error_msg;
	static double svm_c = 1.0, svm_gamma = 1.0 / NUM_FEATURES;
	static char *org_dir = NULL, *dec_dir = NULL, *modelfile = NULL /**modelfile_harris*/;
	
	/*******************Harris******************/
	HARRIS *harris;
	HARRIS *harris_list[MAX_IMAGE];
	/*******************************************/

	cpu_time();
	setbuf(stdout, 0);
	for (i = 1; i < argc; i++)
	{
		if (argv[i][0] == '-')
		{
			switch (argv[i][1])
			{
			case 'L':
				num_class = atoi(argv[++i]);
				if (num_class < 3 || num_class > MAX_CLASS || (num_class % 2) == 0)
				{
					fprintf(stderr, "# of classes is wrong!\n");
					exit(1);
				}
				break;
			case 'C':
				svm_c = atof(argv[++i]);
				break;
			case 'G':
				svm_gamma = atof(argv[++i]);
				break;
			case 'S':
				sig_gain = atof(argv[++i]);
				break;
			default:
				fprintf(stderr, "Unknown option: %s!\n", argv[i]);
				exit(1);
			}
		}
		else
		{
			if (org_dir == NULL)
			{
				org_dir = argv[i];
			}
			else if (dec_dir == NULL)
			{
				dec_dir = argv[i];
			}
			else
			{
				modelfile = argv[i];
			}
		}
	}
	if (modelfile == NULL)
	{
#ifdef LEAVE_ONE_OUT
		printf("Usage: %s [options] original_dir decoded_dir model.svm\n",
#else
		printf("Usage: %s [options] original.pgm decoded.pgm model.svm\n",
#endif
			   argv[0]);
		printf("    -L num  The number of classes [%d]\n", num_class);
		printf("    -C num  Penalty parameter for SVM [%f]\n", svm_c);
		printf("    -G num  Gamma parameter for SVM [%f]\n", svm_gamma);
		printf("    -S num  Gain factor for sigmoid function [%f]\n", sig_gain);
		exit(0);
	}

	//svm_learning start
	num_img = set_images(org_dir, dec_dir, oimg_list, dimg_list);

	/********************************************/
	/*				  Harris					*/
	/*******************************************/
	harris = (HARRIS *)calloc(1, sizeof(HARRIS));
	set_harris(harris, harris_list, oimg_list, num_img);
	/*******************************************/

	set_thresholds_flat_region_harris(oimg_list, dimg_list, num_img, num_class, th_list, harris_list);
	printf("Number of classes = %d\n", num_class);
	printf("Number of training images = %d\n", num_img);
	printf("Thresholds = {%.1f", th_list[0]);
	for (k = 1; k < num_class / 2; k++)
	{
		printf(", %.1f", th_list[k]);
	}
	printf("}\n");
	printf("Gain factor = %f\n", sig_gain);
	printf("SVM(gamma, C) = (%f,%f)\n", svm_gamma, svm_c);

	elements = 0;
	prob.l = 0;
	srand48(RND_SEED); //drand48()のための初期化
	for (img = 0; img < num_img; img++)
	{
		org = oimg_list[img];
		dec = dimg_list[img];
		harris = harris_list[img];
		for (i = 0; i < dec->height; i++)
		{
			for (j = 0; j < dec->width; j++)
			{
				if (drand48() < SAMPLE_RATIO)
				{
					if (harris->bool_harris[i][j] == 0)
					{
						elements += get_fvector(dec, i, j, sig_gain, fvector);
						prob.l++;
					}
				}
			}
		}
	}
	printf("Number of samples = %d (%d)\n", prob.l, (int)elements);

	/* Setting for LIBSVM */
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;		 /* for poly */
	param.gamma = svm_gamma; /* for poly/rbf/sigmoid */
	param.coef0 = 0;		 /* for poly/sigmoid */

	/* these are for training only */
	param.nu = 0.5;			   /* for NU_SVC, ONE_CLASS, and NU_SVR */
	param.cache_size = 100;	/* in MB */
	param.C = svm_c;		   /* for C_SVC, EPSILON_SVR and NU_SVR */
	param.eps = 1e-3;		   /* stopping criteria */
	param.p = 0.1;			   /* for EPSILON_SVR */
	param.shrinking = 0;	   // Changed /* use the shrinking heuristics */
	param.probability = 0;	 /* do probability estimates */
	param.nr_weight = 0;	   /* for C_SVC */
	param.weight_label = NULL; /* for C_SVC */
	param.weight = NULL;	   /* for C_SVC */
	elements += prob.l;
	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct svm_node *, prob.l);
	x_space = Malloc(struct svm_node, elements);
	for (k = 0; k < num_class; k++)
		cls[k] = 0;
	m = n = 0;
	srand48(RND_SEED);

	for (img = 0; img < num_img; img++)
	{
		org = oimg_list[img];
		dec = dimg_list[img];
		harris = harris_list[img];
		for (i = 0; i < dec->height; i++)
		{
			for (j = 0; j < dec->width; j++)
			{
				if (drand48() < SAMPLE_RATIO)
				{
					if (harris->bool_harris[i][j] == 0)
					{
						label = get_label(org, dec, i, j, num_class, th_list);
						cls[label]++;
						prob.y[m] = label;
						prob.x[m] = &x_space[n];
						get_fvector(dec, i, j, sig_gain, fvector);
						for (k = 0; k < NUM_FEATURES; k++)
						{
							if (fvector[k] != 0.0)
							{
								x_space[n].index = k + 1;
								x_space[n].value = fvector[k];
								n++;
							}
						}
						x_space[n++].index = -1;
						m++;
					}
				}
			}
		}
	}
	for (k = 0; k < num_class; k++)
	{
		printf("CLASS[%d] = %d\n", k, cls[k]);
	}
	error_msg = svm_check_parameter(&prob, &param);
	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}
	model = svm_train(&prob, &param);
	if (svm_save_model(modelfile, model))
	{
		fprintf(stderr, "Can't save model to file %s\n", modelfile);
		exit(1);
	}
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	printf("cpu time: %.2f sec.\n", cpu_time());
	return (0);
}