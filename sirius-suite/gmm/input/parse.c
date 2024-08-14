#include <stdio.h>

void main() {

   FILE *fp, *fp2;
   char buff[255];

   fp = fopen("gmm_data.txt", "r");
   fp2 = fopen("gmm_2560.txt", "w");

   for (int i=0; i<2560*32*32; i++){ 
      fscanf(fp, "%s", buff);
      fputs(buff, fp2);
      fputs(" ",fp2);
   }

   for (int i=0; i<2560*32*32; i++){
      fscanf(fp, "%s", buff);
   }

   fputs("\n",fp2);

   for (int i=0; i<2560*32*32; i++){
      fscanf(fp, "%s", buff);
      fputs(buff, fp2);
      fputs(" ", fp2);
   }

   for (int i=0; i<2560*32*32; i++){
      fscanf(fp, "%s", buff);
   }

   fputs("\n", fp2);

   for (int i=0; i<2560*32; i++){
      fscanf(fp, "%s", buff);
      fputs(buff, fp2);
      fputs(" ", fp2);
   }

   for (int i=0; i<2560*32; i++){
      fscanf(fp, "%s", buff);
   }
   
   fputs("\n", fp2);

   for (int i=0; i<2560*32; i++){
      fscanf(fp, "%s", buff);
      fputs(buff, fp2);
      fputs(" ", fp2);
   }

   for (int i=0; i<2560*32; i++){
      fscanf(fp, "%s", buff);
   }
 
   fclose(fp);
   fclose(fp2);

}
