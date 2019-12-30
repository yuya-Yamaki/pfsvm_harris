#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "svm.h"
#include "pfsvm.h"
struct svm_model *model;
struct svm_node *x;
/*************harris*************/
struct svm_model *model_harris;
struct svm_node *x_harris;
/*******************************/

/*pfsvmによって得られた画像を評価するためのプログラム（PSNRetc...）*/

int main(int argc, char **argv)
{
    IMAGE *org, *dec, *cls;
    int i, j, k, n, label, success;
    int num_class, side_info;
    double th_list[MAX_CLASS / 2], fvector[NUM_FEATURES], sig_gain = 1.0;
    double offset[MAX_CLASS];
    int cls_hist[MAX_CLASS];
    double sn_before, sn_after;
    static char *orgimg = NULL, *decimg = NULL, *modelfile = NULL, *modimg = NULL;

    /****************harris*****************/
    HARRIS *harris;
    HARRIS *harris_list[1];
    double th_list_harris[MAX_CLASS];
    static char *modelfile_harris = NULL;
    double offset_harris[MAX_CLASS];
    int cls_hist_harris[MAX_CLASS];
    double fvector_harris[NUM_FEATURES];
    int m, t;
    /***************************************/

    cpu_time();
    setbuf(stdout, 0);
    for (i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            switch (argv[i][1])
            {
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
            if (orgimg == NULL)
            {
                orgimg = argv[i];
            }
            else if (decimg == NULL)
            {
                decimg = argv[i];
            }
            else if (modelfile == NULL)
            {
                modelfile = argv[i];
            }
            else if (modelfile_harris == NULL)
            {
                modelfile_harris = argv[i];
            }
            else
            {
                modimg = argv[i];
            }
        }
    }
    if (modimg == NULL)
    {
        printf("Usage: %s [option] original.pgm decoded.pgm model.svm model_harris.svm modified.pgm\n",
               argv[0]);
        printf("    -S num  Gain factor for sigmoid-like function[%f]\n", sig_gain);
        exit(0);
    }
    /*orgは原画像,decは再生画像*/
    org = read_pgm(orgimg);
    dec = read_pgm(decimg);
    cls = alloc_image(org->width, org->height, 255);
    if ((model = svm_load_model(modelfile)) == 0 || (model_harris = svm_load_model(modelfile_harris)) == 0 )
    {
        fprintf(stderr, "can't open model file %s %s\n", modelfile, modelfile_harris);
        exit(1);
    }

    /*******************harris********************/
    harris = (HARRIS *)calloc(1, sizeof(HARRIS));
    //set_harris(harris, harris_list, &org, 1);
    set_harris_for_check(harris, harris_list, &dec, 1);
    harris = harris_list[0];
    /********************************************/

    /*閾値の取得*/
    num_class = model->nr_class;
    /*何パーセント正確をはかるためにあるプログラムで実際いらない部分***************************/
    set_thresholds_harris(&org, &dec, 1, num_class, th_list, th_list_harris, harris_list);
    printf("PSNR = %.2f (dB)\n", sn_before = calc_snr(org, dec));
    printf("# of classes = %d\n", num_class);
    printf("Thresholds = {%.1f", th_list[0]);
    for (k = 1; k < num_class / 2; k++)
    {
        printf(", %.1f", th_list[k]);
    }
    printf("}\n");
    printf("Thresholds = {%.1f", th_list_harris[0]);
    for (k = 1; k < num_class / 2; k++)
    {
        printf(", %.1f", th_list_harris[k]);
    }
    printf("}\n");
    printf("Gain factor = %f\n", sig_gain);
    x = Malloc(struct svm_node, NUM_FEATURES + 1);
    x_harris = Malloc(struct svm_node, NUM_FEATURES + 1);
    success = 0;
    for (k = 0; k < num_class; k++)
    {
        offset[k] = 0.0;
        offset_harris[k] = 0.0;
        cls_hist[k] = 0;
        cls_hist_harris[k] = 0;
    }
    /***************************************************************************************/

    /*特徴ベクトルの取得,SVMの入力となる．この特徴ベクトルを用いてどこに分類されるかを機械学習で予測する*/
    for (i = 0; i < dec->height; i++)
    {
        for (j = 0; j < dec->width; j++)
        {
            if (harris->bool_harris[i][j] == 0)
            {
                get_fvector(dec, i, j, sig_gain, fvector);
                n = 0;
                for (k = 0; k < NUM_FEATURES; k++)
                {
                    if (fvector[k] != 0.0)
                    {
                        x[n].index = k + 1;
                        x[n].value = fvector[k];
                        n++;
                    }
                }
                /*各画素を三つのクラスに分類したので，その各画素にラベルを割り当てている*/
                x[n].index = -1;
                label = (int)svm_predict(model, x);
                if (label == get_label(org, dec, i, j, num_class, th_list))
                {
                    success++;
                }
                cls->val[i][j] = label;
                /*各分類の再生誤差の総計を求めている．あとでそのクラスに割り当てられた画素数でわり平均値をオフセット値とする*/
                offset[label] += org->val[i][j] - dec->val[i][j];
                cls_hist[label]++;
            }
            else if(harris->bool_harris[i][j] == 1)
            {
              get_fvector(dec, i, j, sig_gain, fvector_harris);
              t = 0;
              for(k = 0; k < NUM_FEATURES; k++)
              {
                if(fvector_harris[k] != 0.0)
                {
                  x_harris[t].index = k + 1;
                  x_harris[t].value = fvector_harris[k];
                  t++;
                }
              }
              x_harris[t].index = -1;
              label = (int)svm_predict(model_harris, x_harris);
              if(label == get_label(org, dec, i, j, num_class, th_list_harris))
              {
                success++;
              }
              cls->val[i][j] = label;
              offset_harris[label] += org->val[i][j] - dec->val[i][j];
              cls_hist_harris[label]++;
            }
        }
        fprintf(stderr, ".");
    }

    fprintf(stderr, "\n");
    printf("Accuracy = %.2f (%%)\n", 100.0 * success / (dec->width * dec->height));
    side_info = 0;
    for (k = 0; k < num_class; k++)
    {
        if (cls_hist[k] > 0)
        {
            offset[k] /= cls_hist[k]; /*offset値は各クラスの再生誤差の平均値*/
        }
        if(cls_hist_harris > 0)
        {
          offset_harris[k] /= cls_hist_harris[k];
        }
        printf("Offset[%d] = %.2f (%d)\n", k, offset[k], cls_hist[k]);
        printf("Offset[%d] = %.2f (%d)\n", k, offset_harris[k], cls_hist_harris[k]);
        offset[k] = n = floor(offset[k] + 0.5);
        offset_harris[k] = m = floor(offset_harris[k] + 0.5);
        if (n < 0)n = -n;
        if(m < 0)m = -m;
        side_info += ((n + 1) + (m + 1)); // unary code
        if (n > 0)
            side_info++; // sign bit
        if(m > 0)
            side_info++;
    }
    /*各画素にオフセット値を加算する************************************************/
    for (i = 0; i < dec->height; i++)
    {
        for (j = 0; j < dec->width; j++)
        {
            if (harris->bool_harris[i][j] == 0)
            {
                label = cls->val[i][j];
                k = dec->val[i][j] + offset[label]; /*オフセット値の加算*/
                if (k < 0)
                    k = 0;
                if (k > 255)
                    k = 255;
                dec->val[i][j] = k;
                /**************************************************************************/
            }
            else if(harris->bool_harris[i][j] == 1)
            {
              label = cls->val[i][j];
              k = dec->val[i][j] + offset_harris[label];
              if (k < 0)
                  k = 0;
              if (k > 255)
                  k = 255;
              dec->val[i][j] = k;
            }
        }
    }
    printf("PSNR = %.3f (dB)\n", sn_after = calc_snr(org, dec)); /*pfsvmによる輝度補償の時のPSNR*/
    printf("GAIN = %+.3f (dB)\n", sn_after - sn_before);
    printf("SIDE_INFO = %d (bits)\n", side_info);
    write_pgm(dec, modimg);
    svm_free_and_destroy_model(&model);
    svm_free_and_destroy_model(&model_harris);
    free(x);
    free(x_harris);
    printf("cpu time: %.2f sec.\n", cpu_time());
    return (0);
}
