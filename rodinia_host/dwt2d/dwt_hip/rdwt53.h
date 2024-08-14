#ifndef RDWT53_H_
#define RDWT53_H_

void rdwt53(int * in, int * out, int sizeX, int sizeY, int levels);
template <int WIN_SX, int WIN_SY>
void launchRDWT53Kernel (int * in, int * out, const int sx, const int sy);

#endif
