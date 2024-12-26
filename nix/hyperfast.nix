{inputs, ...}: {
  imports = [inputs.treefmt-nix.flakeModule inputs.pre-commit-hooks.flakeModule];
  perSystem = {config, ...}: {
    treefmt.config.programs = {
      mypy = {
        enable = true;
        directories = {
          "hyperfast".modules = [
            "src"
          ];
        };
      };
      ruff-check.enable = true;
      ruff-format.enable = true;
    };

    # Enable pre-commit hook of the treefmt declared before
    pre-commit = {
      settings = {
        hooks = {
          treefmt = {
            enable = true;
            pass_filenames = false;
          };
        };
      };
    };
  };
}
