{inputs, ...}: {
  perSystem = {
    config,
    system,
    ...
  }: let
    pkgs = import inputs.nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
  in {
    devShells.default = pkgs.mkShell {
      name = "hyperfast";
      buildInputs = with pkgs; [
        git
        gitRepo
        gnupg
        autoconf
        curl
        procps
        gnumake
        util-linux
        m4
        gperf
        unzip
        cudatoolkit
        linuxPackages.nvidia_x11
        libGLU
        libGL
        xorg.libXi
        xorg.libXmu
        freeglut
        xorg.libXext
        xorg.libX11
        xorg.libXv
        xorg.libXrandr
        zlib
        ncurses5
        stdenv.cc
        binutils
        uv
      ];
      shellHook = ''
        export CUDA_PATH=${pkgs.cudatoolkit}
        export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
        export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
        export EXTRA_CCFLAGS="-I/usr/include"
        # Yeah it is what it is I have a 1080
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      '';
    };
  };
}
