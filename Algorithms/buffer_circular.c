#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define SPARSITY  48
#define BUFFER_SIZE 2016
#define STEP BUFFER_SIZE/SPARSITY
int main(){
    
    int buffer [BUFFER_SIZE] = {0};
    int exit   [STEP] = {0};
    int * w_ptr = buffer;
    int * r_ptr = buffer;
    int diff_w = 0;
    int diff_r = 0;
    for (int i = 0; i < BUFFER_SIZE ;i++){
        memcpy(w_ptr,&i,sizeof(i));
        w_ptr +=STEP;
        
        diff_w =  w_ptr - &buffer[BUFFER_SIZE -1];
        if (diff_w > 0){
            w_ptr = buffer + diff_w;
        }  
    }
    
    printf("\n\n");
    for (int j = 0; j < SPARSITY ; j ++){
        *w_ptr = 24309 - j;
        memcpy(exit,r_ptr,sizeof(exit));
        for (int i = 0 ; i < STEP ; i++){
            printf("%d,",exit[i]);
        }
        printf("\n");
        r_ptr+=STEP;
        w_ptr+=STEP;        
    }

    return -1;
}