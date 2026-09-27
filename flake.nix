{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-unstable";
    };
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      python = pkgs.python3;
      pythonPkgs = python.pkgs;

      dependencies = with pythonPkgs; [
        catboost
        folium
        geopandas
        kagglehub
        matplotlib
        numpy
        optuna
        pandas
        pandas-stubs
        plotly
        scikit-learn
        seaborn
        statsmodels
        types-tqdm
        xgboost

        click
        rich
        typer
      ];

      devDependencies = with pythonPkgs; [
        python
        ipykernel
        ipython
        ipywidgets
        jupyterlab

        pkgs.bashInteractive # Without this there would be broken console when using direnv.
      ];

      lintDependencies = with pkgs; [
        # Format & lint tools.
        git
        gitleaks
        markdownlint-cli2
        (mdformat.withPlugins (
          ps: with ps; [
            mdformat-beautysh
            mdformat-gfm
          ]
        ))
        mypy
        nixfmt
        prettier
        ruff
        shellcheck
        shfmt
        statix
        stylua
        typos
        yamllint
      ];

      repo = pkgs.writeShellApplication {
        name = "repo";
        runtimeInputs = [ (python.withPackages (_: dependencies)) ] ++ lintDependencies;
        text = ''
          export IN_NIX_SHELL=impure
          cd "$(git rev-parse --show-toplevel)"
          exec python3 ./repo.py "$@"
        '';
      };

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = dependencies ++ devDependencies ++ lintDependencies ++ [ repo ];
        shellHook = ''
          export IN_NIX_SHELL=impure
          export JUPYTER_PATH="${pythonPkgs.jupyterlab-widgets}/share/jupyter''${JUPYTER_PATH:+:$JUPYTER_PATH}"
          echo "Welcome to the project devshell!"
        '';
      };

      packages.${system} = { inherit repo; };

      apps.${system}.default = {
        type = "app";
        program = pkgs.lib.getExe repo;
      };
    };
}
