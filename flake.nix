{
  description = "Serve a static website using Docker.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        siteDir = ./static;
        sws = pkgs.static-web-server;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ sws ];
          shellHook = ''
            echo "static-web-server is available. You can start it with:"
            echo "  static-web-server --root ${siteDir}"
          '';
        };

        packages.default = pkgs.dockerTools.buildImage {
          name = "static-web-server-site";
          tag = "latest";
          contents = [ sws siteDir ];
          config = {
            Cmd = [ "${sws}/bin/static-web-server" "--root" "${siteDir}" ];
            ExposedPorts = { "80/tcp" = {}; };
          };
        };

        apps.default = flake-utils.lib.mkApp {
          drv = pkgs.writeShellApplication {
            name = "serve-static";
            text = ''
              exec ${sws}/bin/static-web-server --root ${siteDir}
            '';
          };
        };
      });
}
