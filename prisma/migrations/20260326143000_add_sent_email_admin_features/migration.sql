-- AlterTable (idempotent for DBs already synced via db push)
ALTER TABLE "OutboundEmail" ADD COLUMN IF NOT EXISTS "fromEmail" TEXT;
ALTER TABLE "OutboundEmail" ADD COLUMN IF NOT EXISTS "submissionId" TEXT;
ALTER TABLE "OutboundEmail" ADD COLUMN IF NOT EXISTS "toName" TEXT;
ALTER TABLE "OutboundEmail" ALTER COLUMN "status" SET DEFAULT 'sent';

-- CreateIndex
CREATE INDEX IF NOT EXISTS "OutboundEmail_submissionId_createdAt_idx" ON "OutboundEmail"("submissionId", "createdAt");

-- AddForeignKey (skip if constraint already exists)
DO $outer$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint c
    INNER JOIN pg_class t ON c.conrelid = t.oid
    WHERE c.conname = 'OutboundEmail_submissionId_fkey'
      AND t.relname = 'OutboundEmail'
  ) THEN
    ALTER TABLE "OutboundEmail" ADD CONSTRAINT "OutboundEmail_submissionId_fkey"
      FOREIGN KEY ("submissionId") REFERENCES "ContactSubmission"("id")
      ON DELETE SET NULL ON UPDATE CASCADE;
  END IF;
END
$outer$;
