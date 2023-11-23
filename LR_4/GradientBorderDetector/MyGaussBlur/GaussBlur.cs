﻿namespace MyGaussBlur
{
    public class GaussBlur
    {
        private static double MyGaussFunc(int x, int y, int size, double sigma)
        {
            int a = size / 2 + 1;
            int b = size / 2 + 1;
            return (1 / ((2 * Math.PI) * Math.Pow(sigma, 2))) * Math.Exp(-((Math.Pow((x - a), 2) + Math.Pow((y - b), 2)) / (2 * Math.Pow(sigma, 2))));
        }

        private static void MatrixNorm(List<List<double>> matrix)
        {
            double sum = 0;
            for (int i = 0; i < matrix.Count(); i++)
            {
                for (int j = 0; j < matrix[i].Count(); j++)
                {
                    sum += matrix[i][j];
                }
            }

            for (int i = 0; i < matrix.Count(); i++)
            {
                for (int j = 0; j < matrix[i].Count(); j++)
                {
                    matrix[i][j] = matrix[i][j] / sum;
                }
            }
        }

        private static List<List<double>> FillGaussMatrix(int size, double sigma)
        {
            List<List<double>> matrix = new();
            for (int i = 0; i < size; i++)
            {
                matrix.Add(new List<double>());
                for (int j = 0; j < size; j++)
                {
                    matrix[i].Add(MyGaussFunc(i, j, size, sigma));
                }
            }
            MatrixNorm(matrix);

            return matrix;
        }

        public static List<List<int>> GaussianBlur(List<List<int>> img, int gaussMatrixSize, double sigma)
        {
            int height = img.Count();
            int width = img[0].Count();

            int x_start = gaussMatrixSize / 2;
            int x_end = height - gaussMatrixSize / 2;
            int y_start = gaussMatrixSize / 2;
            int y_end = width - gaussMatrixSize / 2;

            List<List<double>> gaussMatrix = FillGaussMatrix(gaussMatrixSize, sigma);
            List<List<int>> blurMatrix = new(img);


            for (int i = x_start; i < x_end; i++)
            {
                for (int j = y_start; j < y_end; j++)
                {
                    double value = 0;
                    int ii = 0;
                    for (int k = i - gaussMatrixSize / 2; k < i + gaussMatrixSize / 2; k++)
                    {
                        int jj = 0;
                        for (int c = j - gaussMatrixSize / 2; c < j + gaussMatrixSize / 2; c++)
                        {
                            value += img[k][c] * gaussMatrix[ii][jj];
                            jj++;
                        }
                        ii++;
                    }
                    blurMatrix[i][j] = Convert.ToInt32(value);
                }
            }
            return blurMatrix;
        }

    }
}