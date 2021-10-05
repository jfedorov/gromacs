#!/usr/bin/env bash
#
# Build and install CP2K with linkable library for use by GROMACS.
# See GROMACS instructions (link TBD) for user installation.

apt-get update
apt-get --assume-yes install gfortran

# Download CP2K 8.1 release from GitHub
git clone --recursive -b support/v8.1 https://github.com/cp2k/cp2k.git $CP2K_DIR

# Install minimal toolchain
cd $CP2K_DIR/tools/toolchain

./install_cp2k_toolchain.sh -j 4 --mpi-mode=no --with-libxsmm=no --with-elpa=no --with-libxc=no \
--with-libint=no --with-gsl=no --with-libvdwxc=no --with-spglib=no --with-hdf5=no \
--with-spfft=no --with-cosma=no  --with-libvori=no --with-sirius=no --with-fftw=system 

# Copy ARCH file
cp install/arch/local.ssmp $CP2K_DIR/arch/

# Make libcp2k
cd $CP2K_DIR
make -j 4 ARCH=local VERSION=ssmp libcp2k
