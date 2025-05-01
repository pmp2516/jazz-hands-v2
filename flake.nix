{
  description = "Jazz Hands development environments, dependency closure, and application package.";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/master";
    uv2nix.url = "github:/adisbladis/uv2nix";
    uv2nix.inputs.nixpkgs.follows = "nixpkgs";
    uv2nix_hammer_overrides.url = "github:TyberiusPrime/uv2nix_hammer_overrides";
    uv2nix_hammer_overrides.inputs.nixpkgs.follows = "nixpkgs";
  };
  outputs =
    {
      nixpkgs,
      uv2nix,
      uv2nix_hammer_overrides,
      ...
    }:
    let
      inherit (uv2nix.inputs) pyproject-nix;
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };

      # Generate overlay
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      pyprojectOverrides = pkgs.lib.composeExtensions (uv2nix_hammer_overrides.overrides pkgs) (
        # use uv2nix_hammer_overrides.overrides_debug
        #   to see which versions were matched to which overrides
        #  use uv2nix_hammer_overrides.overrides_strict / overrides_strict_debug
        #  to use only overrides exactly matching your python package versions

        _final: _prev: {
          # place additional overlays here.
          #a_pkg = prev.a_pkg.overrideAttrs (old: nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.someBuildTool] ++ (final.resolveBuildSystems { setuptools = [];});
        }
      );
      interpreter = pkgs.python312;

      # Construct package set
      pythonSet' =
        (pkgs.callPackage pyproject-nix.build.packages { python = interpreter; }).overrideScope
          overlay;

      # Override host packages with build fixups
      pythonSet = pythonSet'.pythonPkgsHostHost.overrideScope pyprojectOverrides;
      venv = pythonSet.mkVirtualEnv "jazz-hands-venv" workspace.deps.default;
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = [
          pkgs.uv
          venv
        ];
      };
      devShells.x86_64-linux.uv = pkgs.mkShell { packages = [ nixpkgs.legacyPackages.x86_64-linux.uv ]; };
    };
}
