#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "pfsvm.h"

/******************************************************************************************************************/
/******************************************************************************************************************/
/******************************************************************************************************************/
/*                                                                                                                */
/*                                                                                                                */
/*                                                                                                                */
/*                                             Harris corner detection                                            */
/*                                                                                                                */
/*                                                                                                                */
/*                                                                                                                */
/******************************************************************************************************************/
/******************************************************************************************************************/
/******************************************************************************************************************/
HARRIS *alloc_harris(int height, int width)
{
    HARRIS *harris;
    int i;

    harris = (HARRIS *)calloc(1, sizeof(HARRIS));
    harris->bool_harris = (int **)calloc(height, sizeof(int));
    for (i = 0; i < height; i++)
    {
        harris->bool_harris[i] = (int *)calloc(width, sizeof(int));
    }

    return harris;
}

IMAGE *lowpassGauss_org(IMAGE *org, double *fil, int height, int width, int maxval)
{
    IMAGE *mod;
    int i, j, s, t, x, y;
    int tmp = 0, z = 0;

    mod = alloc_image(height, width, maxval);

    //printf("lowpassGauss_org function start\n");
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            for (s = -1; s <= 1; s++)
            {
                for (t = -1; t <= 1; t++)
                {
                    x = j + t;
                    if (x < 0)
                        x = 0;
                    else if (x > width - 1)
                        x = width - 1;
                    y = i + s;
                    if (y < 0)
                        y = 0;
                    else if (y > height - 1)
                        y = height - 1;

                    tmp += org->val[y][x] * fil[z];
                    z++;
                }
            }
            mod->val[i][j] = tmp;
            z = 0;
            tmp = 0;
        }
    }
    mod->height = height;
    mod->width = width;
    mod->maxval = maxval;

    //printf("ok! lowpassGauss_org function\n");
    return mod;
}

double **convolve(IMAGE *img, int *fil, int height, int width)
{
    double **mod;
    int i, j, s, t, x, y;
    int tmp = 0, z = 0;

    mod = (double **)calloc(height, sizeof(double *));
    for (i = 0; i < height; i++)
    {
        mod[i] = (double *)calloc(width, sizeof(double));
    }
    //printf("convolve function start\n");
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            for (s = -1; s <= 1; s++)
            {
                for (t = -1; t <= 1; t++)
                {
                    x = j + t;
                    if (x < 0)
                        x = 0;
                    else if (x > width - 1)
                        x = width - 1;
                    y = i + s;
                    if (y < 0)
                        y = 0;
                    else if (y > height - 1)
                        y = height - 1;

                    tmp += img->val[y][x] * fil[z];
                    z++;
                }
            }
            mod[i][j] = tmp;
            z = 0;
            tmp = 0;
        }
    }
    //printf("ok! convolve function\n");
    return mod;
}

double **convolve_Gauss(double **dorg, double *fil, int height, int width)
{
    double **mod;
    int i, j, s, t, x, y;
    int tmp = 0, z = 0;

    mod = (double **)calloc(height, sizeof(double *));
    for (i = 0; i < height; i++)
    {
        mod[i] = (double *)calloc(width, sizeof(double));
    }
    //printf("convolve_Gauss function start\n");
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            for (s = -1; s <= 1; s++)
            {
                for (t = -1; t <= 1; t++)
                {
                    x = j + t;
                    if (x < 0)
                        x = 0;
                    else if (x > width - 1)
                        x = width - 1;
                    y = i + s;
                    if (y < 0)
                        y = 0;
                    else if (y > height - 1)
                        y = height - 1;

                    tmp += dorg[y][x] * fil[z];
                    z++;
                }
            }
            mod[i][j] = tmp;
            z = 0;
            tmp = 0;
        }
    }
    //printf("ok! convolve_Gauss function\n");
    return mod;
}

double **square(double **dorg, int height, int width)
{
    int i, j;
    double **mod;

    mod = (double **)calloc(height, sizeof(double *));
    for (i = 0; i < height; i++)
    {
        mod[i] = (double *)calloc(width, sizeof(double));
    }

    //printf("square function start\n");
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            mod[i][j] = dorg[i][j] * dorg[i][j];
        }
    }

    //printf("ok! square function\n");
    return mod;
}

double **dxdy_calc(double **dx, double **dy, int height, int width)
{
    int i, j;
    double **mod;

    mod = (double **)calloc(height, sizeof(double *));
    for (i = 0; i < height; i++)
    {
        mod[i] = (double *)calloc(width, sizeof(double));
    }

    //printf("dxdy_calc function start\n");
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            mod[i][j] = dx[i][j] * dy[i][j];
        }
    }

    //printf("ok! dxdy_calc function\n");
    return mod;
}

double **harris_calc(double **g_dx2, double **g_dy2, double **g_dxdy, int height, int width)
{
    double **mod;
    int i, j;

    mod = (double **)calloc(height, sizeof(double *));
    for (i = 0; i < height; i++)
    {
        mod[i] = (double *)calloc(width, sizeof(double));
    }

    //printf("harris_calc function start\n");
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            //harris = (AB - C^2) - λ((A+B)^2)
            mod[i][j] = (g_dx2[i][j] * g_dy2[i][j]) - (g_dxdy[i][j] * g_dxdy[i][j]) - lambda * ((g_dx2[i][j] + g_dy2[i][j]) * (g_dx2[i][j] + g_dy2[i][j]));
        }
    }

    //printf("ok! harris_calc function\n");
    return mod;
}

void harris_feature(HARRIS **harris_list, HARRIS *harris, double **harris_R, int img, int height, int width)
{
    int i, j;
    double max, min;

    //printf("harris_feature function start\n");
    max = min = harris_R[0][0];
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            if (max < harris_R[i][j])
                max = harris_R[i][j];
            if (min > harris_R[i][j])
                min = harris_R[i][j];
        }
    }

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            if (harris_R[i][j] >= th_harris * max)
            {
                harris->bool_harris[i][j] = 1;
            }
            else if (harris_R[i][j] <= (-1) * th_harris * max)
            {
                harris->bool_harris[i][j] = 1;
            }
            else
            {
                harris->bool_harris[i][j] = 0;
            }
        }
    }

    harris_list[img] = harris;
    //printf("ok! harris_feature function\n");
}

void set_harris(HARRIS *harris, HARRIS **harris_list, IMAGE **oimg_list, int num_img)
{
    IMAGE *org, *org_lowpass;
    int img;
    int height, width, maxval;
    double **dx, **dy, **dx2, **dy2, **dxdy, **g_dx2, **g_dy2, **g_dxdy;
    double **harris_R;

    //FILE *fp_harris;
    int i;
    org = oimg_list[0];
    height = org->height;
    width = org->width;
    maxval = org->maxval;

    harris->bool_harris = (int **)calloc(height, sizeof(int *));
    for (i = 0; i < height; i++)
    {
        harris->bool_harris[i] = (int *)calloc(width, sizeof(int));
    }

    int sobel_x[9] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};

    int sobel_y[9] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1};

    double Gaussian[9] = {
        0.0625, 0.125, 0.0625,
        0.125, 0.25, 0.125,
        0.0625, 0.125, 0.0625};

    for (img = 0; img < num_img; img++)
    {
        org = oimg_list[img];
        height = org->height;
        width = org->width;
        maxval = org->maxval;
        harris_list[img] = alloc_harris(height, width);

        org_lowpass = lowpassGauss_org(org, Gaussian, height, width, maxval);

        dx = convolve(org_lowpass, sobel_x, height, width);
        dy = convolve(org_lowpass, sobel_y, height, width);

        dx2 = square(dx, height, width);
        dy2 = square(dy, height, width);
        dxdy = dxdy_calc(dx, dy, height, width);

        g_dx2 = convolve_Gauss(dx2, Gaussian, height, width);
        g_dy2 = convolve_Gauss(dy2, Gaussian, height, width);
        g_dxdy = convolve_Gauss(dxdy, Gaussian, height, width);

        harris_R = harris_calc(g_dx2, g_dy2, g_dxdy, height, width);
        harris_feature(harris_list, harris, harris_R, img, height, width);
    }
    //check->start
    // harris = harris_list[0];
    // fp_harris = fopen("harris_check.pgm", "wb");
    // fprintf(fp_harris, "P5\n%d %d\n%d\n", width, height, maxval);
    // for (i = 0; i < height; i++)
    // {
    //     for (j = 0; j < width; j++)
    //     {
    //         putc(harris->bool_harris[i][j] * 255, fp_harris);
    //     }
    // }
    // fclose(fp_harris);
    // return;
    //check->end
}