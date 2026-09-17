# Desktop loading visual acceptance

From the repository root, run in two terminals:

```sh
node desktop/test/loading/server.mjs
node desktop/test/loading/verify.mjs
```

Uses the actual renderer, shared components, fonts, and Tailwind build. The test-only Vite plugin replaces the client observation boundary and renders App with a controlled registry instead of running production boot. It neither starts an inference service nor touches user configurations. None of this fixture is included in the desktop build.

Checks all seven pages at 800, 1120, and 1600 pixels, in light and dark themes. It checks loading, loaded, refresh, and failure transitions; reduced motion; noninteractive placeholders; horizontal overflow; and matching frame geometry. It saves 84 screenshots and measured geometry under ignored `specs/26-09-15/page-skeletons/screenshots/` for visual inspection.

When changing a layout, edit `page-layout.ts` and its content/skeleton slots together, then rerun this check. Usage and memory intentionally render the same figure component for both states. Keep fixture examples representative, including long names and both library and downloadable cards. Unknown row counts, text wrapping, installation states, and error messages cannot be determined before observations arrive: geometry assertions cover stable frames, with a small tolerance for Discover's unknown text. Do not add artificial minimum loading delays or cached copies of server data to make screenshots match.

The skeleton primitive follows [shadcn Skeleton](https://ui.shadcn.com/docs/components/radix/skeleton), adapted to the shared slate palette and reduced motion. Headings and immediately available settings remain visible. Errors never become indefinite skeletons.

For Discover assessment progress, run `node desktop/test/loading/assessment.mjs` with the same fixture server. It checks partial assessment, count updates, withheld recommendations, unchanged panel geometry at completion, errors, and reduced motion across three widths and both themes.

For the Discover download takeover, run `node desktop/test/loading/download.mjs` with the same fixture server. It checks unchanged panel geometry, hidden profile controls, byte counts, transfer speed, ETA, indeterminate progress, cancellation access, and profile restoration at three widths in both themes.
