{
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/nixos-25.11";
    };
    nur = {
      url = "github:nix-community/nur";
      inputs.nixpkgs.follows = "nixpkgs";
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

      pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
      inherit (pyproject) project;

      dependencies = map (
        dep: pythonPkgs.${builtins.head (builtins.split "(==|>=|<=|!=|~=|>|<)" dep)}
      ) project.dependencies;

      supportDependencies = with pythonPkgs; [
        click
        pandas-stubs
        rich
        typer
      ];

      lintDependencies = with pkgs; [
        # Format & lint tools.
        git
        gitleaks
        markdownlint-cli2
        mdformat
        mypy
        nixfmt
        prettier
        pythonPkgs.mdformat-tables
        ruff
        shellcheck
        shfmt
        statix
        stylua
        typos
        yamllint
      ];

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [ (python.withPackages (ps: dependencies ++ supportDependencies)) ] ++ lintDependencies;
        shellHook = ''
          export IN_NIX_SHELL=impure
          echo "Welcome to the project devshell!"
        '';
      };
    };
}
