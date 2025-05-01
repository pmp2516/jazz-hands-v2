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
        appClosure = pkgs.buildEnv {
          name = "jazz-hands-closure";
          paths = [ sws siteDir ];
        };
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
          name = "jazz-hands-static";
          tag = "latest";
          copyToRoot = [ sws siteDir ];
          config = {
            Entrypoint = [ "${sws}/bin/static-web-server" "--root" "${siteDir}" ];
            ExposedPorts = { "80/tcp" = {}; };
          };
        };

        packages.appClosure = appClosure;

        apps.default = flake-utils.lib.mkApp {
          drv = pkgs.writeShellApplication {
            name = "serve-jazz-hands-static";
            text = ''
              exec ${sws}/bin/static-web-server --root ${siteDir}
            '';
          };
        };
      });
}
