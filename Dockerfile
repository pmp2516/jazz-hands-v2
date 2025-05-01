# syntax=docker/dockerfile:1
FROM nixos/nix AS builder

COPY . /src
WORKDIR /src

RUN nix \
    --extra-experimental-features "nix-command flakes" \
    build .#appClosure

RUN mkdir /tmp/store-closure && \
    cp -R $(nix-store -qR result) /tmp/store-closure

FROM scratch

WORKDIR /app

COPY --from=builder /tmp/store-closure /nix/store
COPY --from=builder /src/result /app/result
COPY --from=builder /src/static /app/static

ENTRYPOINT [ "/app/result/bin/static-web-server", "--root", "/app/static" ]

